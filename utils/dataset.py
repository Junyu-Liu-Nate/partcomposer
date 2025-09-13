from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class PartComposerBaseDataset(Dataset):
    """
    Dataset for multiple images (the parent class for PartComposerSynthDataset)
    Contains basic image operations.
    """
    def __init__(
        self,
        instance_data_root,
        placeholder_tokens,
        tokenizer,
        use_bg_tokens=False,
        bg_data_root=None,
        bg_placeholder_tokens=None,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        flip_p=0.5,
        randomize_unused_mask_areas = False
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p
        self.randomize_unused_mask_areas = randomize_unused_mask_areas

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        ### Read and load the instance images and masks
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.placeholder_tokens = placeholder_tokens
        self.all_placeholder_tokens = []
        for tokens_list in self.placeholder_tokens:
            for token in tokens_list:
                self.all_placeholder_tokens.append(token)

        self.instances = []
        for folder in sorted(self.instance_data_root.iterdir()):
            if folder.is_dir():
                instance_img_path = folder / "img.jpg"
                instance_image = self.image_transforms(Image.open(instance_img_path))

                instance_masks = []
                i = 0
                while (folder / f"mask{i}.png").exists():
                    instance_mask_path = folder / f"mask{i}.png"
                    curr_mask = Image.open(instance_mask_path)
                    curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
                    instance_masks.append(curr_mask)
                    i += 1

                if instance_masks:
                    instance_masks = torch.cat(instance_masks)

                self.instances.append((instance_image, instance_masks))

        self._length = len(self.instances)

        ### Read and load the background images
        self.use_bg_tokens = use_bg_tokens
        if self.use_bg_tokens:
            self.bg_data_root = Path(bg_data_root)
            if not self.bg_data_root.exists():
                raise ValueError(f"Background images root {self.bg_data_root} doesn't exist.")

            self.bg_imgs = []
            for bg_img_path in sorted(self.bg_data_root.iterdir()):
                if bg_img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.bg_imgs.append(self.image_transforms(Image.open(bg_img_path)))
            self.bg_placeholder_tokens = bg_placeholder_tokens

        ### Disgarded
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self._length)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

    def __len__(self):
        return self._length

    def form_prompt_chair(self, tokens_ids_to_use, tokens_to_use):
        part_map = ['armrest', 'backrest', 'legs', 'seat']
        prompt = "a photo of a chair with "
        for i, token_id in enumerate(tokens_ids_to_use):
            prompt += f"{tokens_to_use[i]} {part_map[token_id]} and "
        prompt = prompt[:-5]
        return prompt
    
    #####---------- Mask out the unused mask area ----------#####
    def mask_out_unused_area_white(self, instance_image, instance_masks, tokens_ids_to_use):
        '''
        Mask out the entire unmasked and unused area in the instance image
        This is used to enforce the model to prodice white background in the unused area
        '''
        # Combine all masks corresponding to tokens in tokens_ids_to_use
        used_masks = instance_masks[tokens_ids_to_use]
        combined_used_mask = used_masks.any(dim=0, keepdim=True)
        
        # Set all unmasked areas to pure white
        instance_image = torch.where(combined_used_mask, instance_image, torch.ones_like(instance_image))
        
        return instance_image.squeeze(0)
    
    def mask_out_unused_area_original(self, instance_image, instance_masks, tokens_ids_to_use, original_image):
        """
        Instead of turning unused areas white, restore them to the original image pixels.
        """
        used_masks = instance_masks[tokens_ids_to_use]
        combined_used_mask = used_masks.any(dim=0, keepdim=True)
        instance_image = torch.where(combined_used_mask, instance_image, original_image)
        return instance_image.squeeze(0)

    def __getitem__(self, index):
        instance_image, instance_masks = self.instances[index % len(self.instances)]

        # Get the placeholder tokens for the current index
        current_tokens_list = self.placeholder_tokens[index % len(self.placeholder_tokens)]

        # Determine the number of tokens to use
        num_tokens_to_use = random.randrange(1, len(current_tokens_list) + 1)

        # Randomly select the token IDs to use
        tokens_ids_to_use = random.sample(range(len(current_tokens_list)), k=num_tokens_to_use)

        # Retrieve the actual tokens using the selected indices
        tokens_to_use = [current_tokens_list[tkn_id] for tkn_id in tokens_ids_to_use]
        
        prompt = "a photo of a chair with " + " and ".join(tokens_to_use)

        ###------------------- Mask out the unused mask area -------------------###
        if self.randomize_unused_mask_areas:
            instance_image = self.mask_out_unused_mask_area(instance_image, instance_masks, tokens_ids_to_use)
        ###---------------------------------------------------------------------###

        tokens_ids_to_use_global = []
        ### Here the ids are global ids in all placeholder tokens
        for token_to_use in tokens_to_use:
            tokens_ids_to_use_global.append(self.all_placeholder_tokens.index(token_to_use))
        example = {
            "instance_images": instance_image,
            "instance_masks": instance_masks[tokens_ids_to_use],
            "token_ids": torch.tensor(tokens_ids_to_use_global),
            "instance_prompt_ids": self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        }

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

### The main dataset that is being used in the training
class PartComposerSynthDataset(PartComposerBaseDataset):
    """
    Dataset for multiple images
    Return 1 image within original training images and 1 image of synthesized parts
    """
    def __init__(
        self,
        instance_data_root,
        placeholder_tokens,
        tokenizer,
        use_bg_tokens=False,
        bg_data_root=None,
        bg_placeholder_tokens=None,
        use_prior_data=False,
        prior_data_root=None,
        prior_prob=0.1,
        size=512,
        center_crop=False,
        flip_p=0.5,
        randomize_unused_mask_areas = False,
        set_bg_white = False,
        use_all_sythn = False,
        use_all_instance = False,
        subject_name = 'chair',
        sample_type = 'fixed-num',
        synth_type = '4-corner',
        train_detailed_prompt = False,
        sythn_detailed_prompt = False
    ):
        self.subject_name = subject_name

        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p
        
        self.randomize_unused_mask_areas = randomize_unused_mask_areas
        self.set_bg_white = set_bg_white
        
        self.use_all_sythn = use_all_sythn  ### Whether set all 2 imgs in the training batch as randomly syntheized image
        self.use_all_instance = use_all_instance  ### For BaS baseline only
        self.sample_type = sample_type      ### The type of sampling concept combination
        self.synth_type = synth_type        ### The type of image synthesis method
        
        self.sythn_detailed_prompt = sythn_detailed_prompt ### Use detailed prompt for the synthesized image
        self.train_detailed_prompt = train_detailed_prompt ### Use detailed prompt for the training image - the given image

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        ### Read and load the instance images
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.placeholder_tokens = placeholder_tokens
        self.all_placeholder_tokens = []
        for tokens_list in self.placeholder_tokens:
            for token in tokens_list:
                self.all_placeholder_tokens.append(token)

        self.instances = []
        for folder in sorted(self.instance_data_root.iterdir()):
            if folder.is_dir():
                instance_img_path = folder / "img.jpg"
                instance_image = self.image_transforms(Image.open(instance_img_path))

                instance_masks = []
                i = 0
                while (folder / f"mask{i}.png").exists():
                    instance_mask_path = folder / f"mask{i}.png"
                    curr_mask = Image.open(instance_mask_path)
                    curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
                    instance_masks.append(curr_mask)
                    i += 1

                if instance_masks:
                    instance_masks = torch.cat(instance_masks)

                self.instances.append((instance_image, instance_masks))

        self._length = len(self.instances)

        ### Read and load the background images
        self.use_bg_tokens = use_bg_tokens
        if self.use_bg_tokens:
            self.bg_data_root = Path(bg_data_root)
            if not self.bg_data_root.exists():
                raise ValueError(f"Background images root {self.bg_data_root} doesn't exist.")

            self.bg_imgs = []
            for bg_img_path in sorted(self.bg_data_root.iterdir()):
                if bg_img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.bg_imgs.append(self.image_transforms(Image.open(bg_img_path)))
            self.bg_placeholder_tokens = bg_placeholder_tokens

            print(f"Number of background images: {len(self.bg_imgs)}")

        # ----------  PRIOR‑PRESERVATION IMAGES ----------
        self.use_prior_data = use_prior_data
        self.prior_prob = 0.2
        if self.use_prior_data:
            self.prior_imgs = []

            if prior_data_root is not None:
                self.prior_data_root = Path(prior_data_root)
                if not self.prior_data_root.exists():
                    raise ValueError(f"Prior‑data root {self.prior_data_root} does not exist.")
                for p in sorted(self.prior_data_root.iterdir()):
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.prior_imgs.append(self.image_transforms(Image.open(p)))
                if len(self.prior_imgs) == 0:
                    raise ValueError(f"No images found in {self.prior_data_root}.")
            else:
                raise ValueError("`--prior_data_dir` must be provided for prior‑preservation training.")

    #####----- Sampling within train images -----#####
    def uniform_sample_train_img(self, index, bg_token_idx):
        ###------------------- Sampling concept combinations from the original image -------------------###
        instance_image, instance_masks = self.instances[index % len(self.instances)]

        ### Get the placeholder tokens for the current index
        current_tokens_list = self.placeholder_tokens[index % len(self.placeholder_tokens)]
        # print(f'current_tokens_list: {current_tokens_list}')

        ### Determine the number of tokens to use
        num_tokens_to_use = random.randrange(1, len(current_tokens_list) + 1)
        # print(f'num_tokens_to_use: {num_tokens_to_use}')

        ### Randomly select the token IDs to use
        tokens_ids_to_use = random.sample(range(len(current_tokens_list)), k=num_tokens_to_use)
        # print(f'tokens_ids_to_use: {tokens_ids_to_use}')

        ### Retrieve the actual tokens using the selected indices
        tokens_to_use = [current_tokens_list[tkn_id] for tkn_id in tokens_ids_to_use]
        # print(f'tokens_to_use: {tokens_to_use}')
        
        ### Formalize the prompt
        if self.train_detailed_prompt:
            if len(tokens_ids_to_use) < len(self.placeholder_tokens[index % len(self.instances)]):
                prompt = "a photo of a partial " + self.subject_name + " with " + " and ".join(tokens_to_use)
            else:
                prompt = "a photo of a complete " + self.subject_name + " with " + " and ".join(tokens_to_use)
        else:
            prompt = "a photo of a " + self.subject_name + " with " + " and ".join(tokens_to_use)
        if self.use_bg_tokens:
            prompt += ", on a " + self.bg_placeholder_tokens[bg_token_idx] + " background"
        if self.set_bg_white:
            prompt += ", on a simple white background"
        # print(f'prompt: {prompt}')
        
        ### Replace the background if using a chosen background token
        if self.use_bg_tokens and self.bg_imgs:
            bg_img = self.bg_imgs[bg_token_idx]
            # Replace all areas not covered by any mask with the chosen background
            uncovered_area = (instance_masks.sum(dim=0, keepdim=True) == 0)
            instance_image = torch.where(uncovered_area, bg_img, instance_image)

        ### Mask out the unused mask area
        if self.randomize_unused_mask_areas:
            if self.use_bg_tokens:
                instance_image = self.mask_out_unused_area_original(instance_image, instance_masks, tokens_ids_to_use, bg_img)
            elif self.set_bg_white:
                instance_image = self.mask_out_unused_area_white(instance_image, instance_masks, tokens_ids_to_use)
            else:
                instance_image = self.mask_out_unused_area_original(instance_image, instance_masks, tokens_ids_to_use, instance_image)

        tokens_ids_to_use_global = []
        ### Here the ids are global ids in all placeholder tokens
        for token_to_use in tokens_to_use:
            tokens_ids_to_use_global.append(self.all_placeholder_tokens.index(token_to_use))

        return instance_image, instance_masks[tokens_ids_to_use], tokens_ids_to_use_global, prompt
    
    #####----- Samping across train images -----#####
    ###--- Choices of how many concepts and how to sample them ---###
    def complete_random_comb(self, num_sample = 4):
        '''
          Completely randomly sample num_sample elements, ensuring each image has at least 1 element
        '''
        img_id_map = [] # Map from each token to the image to use. e.g. [0, 1, 1, 1] means the first token uses the first image, and the rest use the second image
        synth_mask_ids = [] 
        synth_tokens_to_use = []

        ###----- Sample method 1: completely random -----###
        total_syhth_num = num_sample
        remaining_syhth_num = total_syhth_num
        # print(self.placeholder_tokens)
        for instance_id, tokens_list in enumerate(self.placeholder_tokens):
            if instance_id == len(self.instances) - 1:
                num_tokens_to_use = remaining_syhth_num
            else:
                num_tokens_to_use = random.randrange(1, remaining_syhth_num)
                remaining_syhth_num -= num_tokens_to_use

            tokens_ids_to_use = random.sample(range(len(tokens_list)), k=num_tokens_to_use)
            # print(f'tokens_ids_to_use: {tokens_ids_to_use}')
            
            img_id_map.extend([instance_id] * num_tokens_to_use)
            synth_mask_ids.extend(tokens_ids_to_use)

            tokens_to_use = [tokens_list[tkn_id] for tkn_id in tokens_ids_to_use]
            # print(f'tokens_to_use: {tokens_to_use}')
            synth_tokens_to_use.extend(tokens_to_use)
        ###----------------------------------------------###

        return img_id_map, synth_mask_ids, synth_tokens_to_use
    
    def per_part_random_comb(self):
        '''
          Completely randomly sample num_sample elements, ensuring each image has at least 1 element.
          The sampled 4 parts are all different and compose a complete subject - for armrest, choose 1; for seat, choose 1...
        '''
        img_id_map = [] # Map from each token to the image to use. e.g. [0, 1, 1, 1] means the first token uses the first image, and the rest use the second image
        synth_mask_ids = [] 
        synth_tokens_to_use = []
        
        ###----- Sample method 2: randomly sample but with 4 parts -----###
        # All sublists have the same length
        num_positions = len(self.placeholder_tokens[0])

        # Loop over each position
        for pos in range(num_positions):
            # Collect tokens at the current position from all sublists
            tokens_at_pos = []
            img_ids_at_pos = []
            for instance_id, tokens_list in enumerate(self.placeholder_tokens):
                tokens_at_pos.append(tokens_list[pos])
                img_ids_at_pos.append(instance_id)
            
            # Randomly select one token from the collected tokens
            idx = random.randrange(len(tokens_at_pos))
            sampled_token = tokens_at_pos[idx]
            sampled_img_id = img_ids_at_pos[idx]
            sampled_mask_id = pos  # The position index serves as the mask ID
            
            # Append the sampled token and associated data to the result lists
            synth_tokens_to_use.append(sampled_token)
            img_id_map.append(sampled_img_id)
            synth_mask_ids.append(sampled_mask_id)

        # After collecting one token per position, shuffle the results together
        combined = list(zip(synth_tokens_to_use, img_id_map, synth_mask_ids))
        random.shuffle(combined)
        synth_tokens_to_use[:], img_id_map[:], synth_mask_ids[:] = zip(*combined)
        ###--------------------------------------------------------------###

        return img_id_map, synth_mask_ids, synth_tokens_to_use

    def subject_random_comb(self):
        '''
        Select tokens from 2 chairs, for each chair, randomly select several parts.
        This will be used in 
        '''
        img_id_map = [] # Map from each token to the image to use. e.g. [0, 1, 1, 1] means the first token uses the first image, and the rest use the second image
        synth_mask_ids = [] 
        synth_tokens_to_use = []
        for instance_id, tokens_list in enumerate(self.placeholder_tokens):
            num_tokens_to_use = random.randrange(1, len(tokens_list) + 1)

            tokens_ids_to_use = random.sample(range(len(tokens_list)), k=num_tokens_to_use)
            # print(f'tokens_ids_to_use: {tokens_ids_to_use}')
            
            img_id_map.append([instance_id] * num_tokens_to_use)
            synth_mask_ids.append(tokens_ids_to_use)

            tokens_to_use = [tokens_list[tkn_id] for tkn_id in tokens_ids_to_use]
            # print(f'tokens_to_use: {tokens_to_use}')
            synth_tokens_to_use.append(tokens_to_use)

        # Randomly swap the subjects while maintaining correspondence
        indices = list(range(len(img_id_map)))
        random.shuffle(indices)

        img_id_map = [img_id_map[i] for i in indices]
        synth_mask_ids = [synth_mask_ids[i] for i in indices]
        synth_tokens_to_use = [synth_tokens_to_use[i] for i in indices]

        return img_id_map, synth_mask_ids, synth_tokens_to_use

    def complete_random_comb_random_num(self):
        '''
          Completely randomly sample random num of elements, ensuring each image has at least 1 element
        '''
        img_id_map = [] # Map from each token to the image to use. e.g. [0, 1, 1, 1] means the first token uses the first image, and the rest use the second image
        synth_mask_ids = [] 
        synth_tokens_to_use = []

        ###----- Sample method 1: completely random -----###
        total_syhth_num = random.randrange(2, 9)
        remaining_syhth_num = total_syhth_num
        # print(self.placeholder_tokens)
        for instance_id, tokens_list in enumerate(self.placeholder_tokens):
            if instance_id == len(self.instances) - 1:
                num_tokens_to_use = min(remaining_syhth_num, 4)
            else:
                num_tokens_to_use = random.randrange(1, min(remaining_syhth_num, 5))
                remaining_syhth_num -= num_tokens_to_use

            tokens_ids_to_use = random.sample(range(len(tokens_list)), k=num_tokens_to_use)
            # print(f'tokens_ids_to_use: {tokens_ids_to_use}')
            
            img_id_map.extend([instance_id] * num_tokens_to_use)
            synth_mask_ids.extend(tokens_ids_to_use)

            tokens_to_use = [tokens_list[tkn_id] for tkn_id in tokens_ids_to_use]
            # print(f'tokens_to_use: {tokens_to_use}')
            synth_tokens_to_use.extend(tokens_to_use)
        ###----------------------------------------------###

        return img_id_map, synth_mask_ids, synth_tokens_to_use

    def generate_concept_comb_across_train_imgs(self, bg_token_idx):
        '''
        The main function to generate the concept combination across train images
        This function call different methods to sample the concepts, and compose the prompt
        The output of this function will be used to synthesize the image
        '''
        if self.sample_type == 'fixed-num':
            img_id_map, synth_mask_ids, synth_tokens_to_use = self.complete_random_comb()
        elif self.sample_type == 'random-num':
            img_id_map, synth_mask_ids, synth_tokens_to_use = self.complete_random_comb_random_num()
        elif self.sample_type == 'per-part':
            img_id_map, synth_mask_ids, synth_tokens_to_use = self.per_part_random_comb()
        elif self.sample_type == 'per-subject':
            img_id_map, synth_mask_ids, synth_tokens_to_use = self.subject_random_comb()
            synth_tokens_to_use_flat = [token for sublist in synth_tokens_to_use for token in sublist]
            synth_tokens_to_use = synth_tokens_to_use_flat
        else:
            raise ValueError(f"Unsupported sample_type: {self.sample_type}")
        
        ### Sample bg tokens
        if self.use_bg_tokens:
            bg_token = self.bg_placeholder_tokens[bg_token_idx]

        if not self.sythn_detailed_prompt:
            synth_prompt = "a photo of " + " and ".join(synth_tokens_to_use)
        else:
            synth_prompt = "a photo of randomly placed " + self.subject_name + " components: " + " and ".join(synth_tokens_to_use)

        if self.set_bg_white:
            synth_prompt += ", on a simple white background"
        elif self.use_bg_tokens:
            synth_prompt += ", on a " + bg_token + " background"

        synth_tokens_ids_to_use_global = []
        ### Here the ids are global ids in all placeholder tokens
        for token_to_use in synth_tokens_to_use:
            synth_tokens_ids_to_use_global.append(self.all_placeholder_tokens.index(token_to_use))

        if self.synth_type == 'random-no-overlap':
            return img_id_map, synth_mask_ids, synth_tokens_to_use ### In this situation, need to syntheize the image first and filter unused tokens
        else:
            return img_id_map, synth_mask_ids, synth_tokens_ids_to_use_global, synth_prompt

    ###--- Choices of how to synthesize images ---###
    def synthesize_across_train_imgs_4_corner(self, img_id_map, synth_mask_ids):
        background = torch.ones((3, self.size, self.size))
        masks_list = []
        positions = [
            (0, 0),  # Upper left
            (0, self.size // 2),  # Upper right
            (self.size // 2, 0),  # Lower left
            (self.size // 2, self.size // 2)  # Lower right
        ]
        quad_size = self.size // 2

        for i in range(4):
            instance_id = img_id_map[i]
            mask_id = synth_mask_ids[i]
            y_start, x_start = positions[i]

            instance_image, instance_masks = self.instances[instance_id]
            mask = instance_masks[mask_id]
            mask = mask.squeeze(0)  # Shape [H, W]
            masked_area = instance_image * mask

            # Randomly scale the masked area
            min_scale = 0.5  # Minimum scaling factor
            max_scale = 1.0  # Maximum scaling factor
            scale_factor = random.uniform(min_scale, max_scale)

            # Compute new size
            orig_height, orig_width = mask.shape
            scaled_height = int(orig_height * scale_factor)
            scaled_width = int(orig_width * scale_factor)

            # Ensure dimensions do not exceed quadrant size
            scaled_height = min(scaled_height, quad_size)
            scaled_width = min(scaled_width, quad_size)

            # Resize masked area and mask
            resized_masked_area = transforms.functional.resize(masked_area, [scaled_height, scaled_width])
            mask = mask.unsqueeze(0)
            resized_mask = transforms.functional.resize(mask, [scaled_height, scaled_width])

            # Calculate end positions
            y_end = y_start + scaled_height
            x_end = x_start + scaled_width

            # Place the resized masked area onto the background
            background[:, y_start:y_end, x_start:x_end] = torch.where(
                resized_mask > 0,
                resized_masked_area,
                background[:, y_start:y_end, x_start:x_end]
            )

            # Create mask for this part
            full_mask = torch.zeros((1, self.size, self.size))
            full_mask[0, y_start:y_end, x_start:x_end] = resized_mask

            masks_list.append(full_mask.unsqueeze(0))

        masks_tensor = torch.cat(masks_list, dim=0)
        return background, masks_tensor

    def synthesize_across_train_imgs_2_subject(self, img_id_map, synth_mask_ids):
        '''
        Place 2 train subjects at the left and right of the background
        Mask out the unused mask area
        '''
        background = torch.ones((3, self.size, self.size))
        masks_list = []
        half_size = self.size // 2

        # Define regions for left and right halves
        regions = [
            (0, 0, self.size, half_size),         # Left half: y_start, x_start, y_end, x_end
            (0, half_size, self.size, self.size)  # Right half
        ]

        for i in range(2):  # For each subject
            instance_ids = img_id_map[i]    # Should be the same instance ID repeated
            mask_ids = synth_mask_ids[i]    # List of selected mask IDs for the subject
            y_start, x_start, y_end, x_end = regions[i]

            # Retrieve the instance image and all masks
            instance_id = instance_ids[0]  # All instance_ids in this sublist are the same
            instance_image, instance_masks = self.instances[instance_id]  # instance_image: [3, H, W], instance_masks: [num_masks, 1, H, W]

            # Create a composite mask with selected parts
            selected_masks = instance_masks[mask_ids]  # Shape: [num_selected_masks, 1, H, W]
            composite_mask = selected_masks.sum(dim=0)  # Shape: [1, H, W]; sum over masks
            composite_mask = (composite_mask > 0).float()

            # Mask out unused parts in the instance image
            masked_instance_image = instance_image * composite_mask

            # Determine scaling factor to fit within the region while preserving aspect ratio
            orig_height, orig_width = instance_image.shape[1:]  # H, W
            region_height = y_end - y_start
            region_width = x_end - x_start

            max_scale_height = region_height / orig_height
            max_scale_width = region_width / orig_width
            max_possible_scale = min(max_scale_height, max_scale_width, 1.0)

            min_scale = 0.5  # Minimum scaling factor
            max_scale = min(max_possible_scale, 1.0)
            if min_scale > max_scale:
                min_scale = max_scale  # Adjust min_scale if necessary

            scale_factor = random.uniform(min_scale, max_scale)

            # Scale preserving aspect ratio
            scaled_height = int(orig_height * scale_factor)
            scaled_width = int(orig_width * scale_factor)

            # Resize masked instance image
            resized_masked_image = transforms.functional.resize(masked_instance_image, [scaled_height, scaled_width])

            # Randomly choose position within the region
            max_y_offset = region_height - scaled_height
            max_x_offset = region_width - scaled_width
            y_offset = random.randint(0, max(0, max_y_offset))
            x_offset = random.randint(0, max(0, max_x_offset))
            y_pos = y_start + y_offset
            x_pos = x_start + x_offset
            y_end_pos = y_pos + scaled_height
            x_end_pos = x_pos + scaled_width

            # Place the resized masked instance image onto the background
            background[:, y_pos:y_end_pos, x_pos:x_end_pos] = torch.where(
                resized_masked_image != 0,
                resized_masked_image,
                background[:, y_pos:y_end_pos, x_pos:x_end_pos]
            )

            # For each selected mask, process individually
            for mask_id in mask_ids:
                mask = instance_masks[mask_id]  # Shape: [1, H, W]
                # Resize mask
                resized_mask = transforms.functional.resize(mask, [scaled_height, scaled_width])

                # Place the resized mask into a full-sized mask
                full_mask = torch.zeros((1, self.size, self.size))
                full_mask[0, y_pos:y_end_pos, x_pos:x_end_pos] = resized_mask[0]

                masks_list.append(full_mask.unsqueeze(0))

        masks_tensor = torch.cat(masks_list, dim=0)  # Shape: [num_parts, 1, H, W]
        return background, masks_tensor

    def rectify_sampled_concepts(self, placed_img_id_map, placed_synth_mask_ids, placed_synth_tokens):
        synth_prompt = "a photo of " + " and ".join(placed_synth_tokens)

        synth_tokens_ids_to_use_global = []
        ### Here the ids are global ids in all placeholder tokens
        for token_to_use in placed_synth_tokens:
            synth_tokens_ids_to_use_global.append(self.all_placeholder_tokens.index(token_to_use))

        return placed_img_id_map, placed_synth_mask_ids, synth_tokens_ids_to_use_global, synth_prompt 
    
    def synthesize_across_train_imgs_random_no_overlap(self, img_id_map, synth_mask_ids, synth_tokens_to_use):
        '''
        Randomly arrange n parts with random scale and position in an image (no overlap)
        '''
        background = torch.ones((3, self.size, self.size))
        masks_list = []
        cumulative_mask = torch.zeros((self.size, self.size), dtype=torch.bool)  # For overlap checking

        num_parts = len(img_id_map)
        max_retries = 10  # Maximum number of retries for placing a part

        # Determine scaling factors based on the number of parts
        # Adjust scaling to fit more parts without overlap
        min_scale = 0.4  # Minimum scaling factor
        max_scale = 1  # Maximum scaling factor
        scale_step = 0.05  # Adjust scaling based on number of parts

        # Adjust max_scale based on the number of parts
        adjusted_max_scale = max(min_scale, max_scale - (num_parts - 1) * scale_step)

        # Lists to keep track of successfully placed parts
        placed_img_id_map = []
        placed_synth_mask_ids = []
        placed_synth_tokens = []
        
        for idx in range(num_parts):
            instance_id = img_id_map[idx]
            mask_id = synth_mask_ids[idx]
            token = synth_tokens_to_use[idx]

            # Retrieve the instance image and mask
            instance_image, instance_masks = self.instances[instance_id]
            mask = instance_masks[mask_id]
            mask = mask.squeeze(0)  # Shape: [H, W]
            masked_area = instance_image * mask

            # Compute original size
            orig_height, orig_width = mask.shape

            # Determine scaling factor
            scale_factor = random.uniform(min_scale, adjusted_max_scale)

            # Compute new size while preserving aspect ratio
            scaled_height = int(orig_height * scale_factor)
            scaled_width = int(orig_width * scale_factor)

            # Resize masked area and mask
            resized_masked_area = transforms.functional.resize(masked_area, [scaled_height, scaled_width])
            resized_mask = transforms.functional.resize(mask.unsqueeze(0), [scaled_height, scaled_width])[0]

            # Try placing the part without overlapping
            placed = False
            for attempt in range(max_retries):
                # Randomly choose position within the image
                max_y_offset = self.size - scaled_height
                max_x_offset = self.size - scaled_width

                if max_y_offset < 0 or max_x_offset < 0:
                    # The scaled part is larger than the image; reduce scale and retry
                    scale_factor *= 0.9  # Reduce scale factor
                    scaled_height = int(orig_height * scale_factor)
                    scaled_width = int(orig_width * scale_factor)
                    resized_masked_area = transforms.functional.resize(masked_area, [scaled_height, scaled_width])
                    resized_mask = transforms.functional.resize(mask.unsqueeze(0), [scaled_height, scaled_width])[0]
                    continue

                y_offset = random.randint(0, max_y_offset)
                x_offset = random.randint(0, max_x_offset)
                y_end_pos = y_offset + scaled_height
                x_end_pos = x_offset + scaled_width

                # Check for overlap
                current_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
                current_mask[y_offset:y_end_pos, x_offset:x_end_pos] = resized_mask > 0

                if not torch.any(cumulative_mask & current_mask):
                    # No overlap; place the part
                    background[:, y_offset:y_end_pos, x_offset:x_end_pos] = torch.where(
                        resized_mask > 0,
                        resized_masked_area,
                        background[:, y_offset:y_end_pos, x_offset:x_end_pos]
                    )

                    # Update cumulative mask
                    cumulative_mask |= current_mask

                    # Create mask for this part
                    full_mask = torch.zeros((1, self.size, self.size))
                    full_mask[0, y_offset:y_end_pos, x_offset:x_end_pos] = resized_mask
                    masks_list.append(full_mask.unsqueeze(0))

                    # Keep track of successfully placed parts
                    placed_img_id_map.append(instance_id)
                    placed_synth_mask_ids.append(mask_id)
                    placed_synth_tokens.append(token)

                    placed = True
                    break  # Exit the retry loop

            if not placed:
                # If unable to place without overlap after max retries, skip this part
                # Remove the part from the lists (handled by not adding to placed_* lists)
                print(f"Warning: Could not place part {idx} ({token}) without overlap after {max_retries} attempts.")

        masks_tensor = torch.cat(masks_list, dim=0) if masks_list else torch.empty(0)

        # Return the updated lists of placed parts
        return background, masks_tensor, placed_img_id_map, placed_synth_mask_ids, placed_synth_tokens

    def synthesize_across_train_imgs_random_overlap(self, img_id_map, synth_mask_ids, bg_token_idx = -1):
        """
        Synth method 4: Pick num of concepts across images, randomly scale and place them in the image.
        Overlapping is possible. Masks are binarized, and out-of-range errors are avoided.
        """
        # background = torch.ones((3, self.size, self.size))  
        ### Use given bg image if learning bg concept
        if bg_token_idx != -1:
            background = self.bg_imgs[bg_token_idx].clone()
        else:
            background = torch.ones((3, self.size, self.size)) # Initialize a white background
        
        combined_mask = torch.zeros((1, self.size, self.size))  # Initialize an empty mask
        masks_list = []
        
        for i in range(len(img_id_map)):
            instance_id = img_id_map[i]
            mask_id = synth_mask_ids[i]

            # Retrieve the instance image and mask
            instance_image, instance_masks = self.instances[instance_id]
            mask = instance_masks[mask_id]  # Shape: [1, H, W]
            masked_area = instance_image * mask

            # Randomly scale the masked area
            min_scale = 0.6  # Minimum scaling factor
            max_scale = 1.0  # Maximum scaling factor
            scale_factor = random.uniform(min_scale, max_scale)

            # Compute new size while preserving aspect ratio
            orig_height, orig_width = mask.shape[1:]  # H, W
            scaled_height = int(orig_height * scale_factor)
            scaled_width = int(orig_width * scale_factor)

            # Resize the masked area and mask
            resized_masked_area = transforms.functional.resize(masked_area, [scaled_height, scaled_width])
            resized_mask = transforms.functional.resize(mask, [scaled_height, scaled_width])

            # Binarize the resized mask to ensure it contains only 0 and 1
            resized_mask = (resized_mask > 0.5).float()

            # Randomly position the resized part within the canvas
            max_y_offset = max(0, self.size - scaled_height)
            max_x_offset = max(0, self.size - scaled_width)
            y_offset = random.randint(0, max_y_offset)
            x_offset = random.randint(0, max_x_offset)
            y_end = y_offset + scaled_height
            x_end = x_offset + scaled_width

            # Adjust overlapping regions in the mask
            overlap_mask = combined_mask[:, y_offset:y_end, x_offset:x_end] > 0
            resized_mask[overlap_mask] = resized_mask[overlap_mask] * (1 - combined_mask[:, y_offset:y_end, x_offset:x_end][overlap_mask])

            # Update the combined mask
            combined_mask[:, y_offset:y_end, x_offset:x_end] += resized_mask
            combined_mask = combined_mask.clamp(0, 1)  # Ensure mask values are in [0, 1]

            # Overlay the resized masked area onto the background
            background[:, y_offset:y_end, x_offset:x_end] = torch.where(
                resized_mask > 0,
                resized_masked_area,
                background[:, y_offset:y_end, x_offset:x_end]
            )

            # Create a full-sized mask for this part
            full_mask = torch.zeros((1, self.size, self.size))
            full_mask[:, y_offset:y_end, x_offset:x_end] = resized_mask
            masks_list.append(full_mask)

        masks_tensor = torch.cat(masks_list, dim=0)  # Stack all masks
        masks_tensor = masks_tensor.unsqueeze(1)  # Add channel dimension
        return background, masks_tensor

    #####----- Synthesize test image -----#####
    def interpret_tokens(self, tokens):
        '''
        Given inference rquired tokens, convert to the corresponding instance ID and mask ID
        The output the same format as concept sampling functions for training: complete_random_comb and complete_random_comb_random_num
        The output is used for synthesizing test images
        '''
        img_id_map = []
        synth_mask_ids = []
        synth_tokens_to_use = tokens

        # For each given token, find which instance and which mask ID it corresponds to
        for t in tokens:
            found = False
            for instance_id, tokens_list in enumerate(self.placeholder_tokens):
                if t in tokens_list:
                    tkn_id = tokens_list.index(t)
                    img_id_map.append(instance_id)
                    synth_mask_ids.append(tkn_id)
                    found = True
                    break
            if not found:
                raise ValueError(f"Token {t} not found in any instance tokens.")

        return img_id_map, synth_mask_ids, synth_tokens_to_use

    def synthesize_test_img(self, tokens):
        img_id_map, synth_mask_ids, synth_tokens_to_use = self.interpret_tokens(tokens)

        synth_image, synth_masks = self.synthesize_across_train_imgs_4_corner(img_id_map, synth_mask_ids)
        # synth_image, synth_masks = self.synthesize_across_train_imgs_random_overlap(img_id_map, synth_mask_ids)
        # synth_image, synth_masks, _, _, _ = self.synthesize_across_train_imgs_random_no_overlap(img_id_map, synth_mask_ids, synth_tokens_to_use)

        return synth_image, synth_masks

    #####----- Visualiztion: used for debug -----#####
    def visualize_instance_imgs(self, instance_image, instance_masks, index):
        print(f"instance_image shape: {instance_image.shape}")
        #----- Save the synthesized image and masks for visualization -----#
        # Unnormalize the image tensor if it's normalized to [-1, 1]
        instance_image_vis = (instance_image + 1) / 2
        instance_image_vis = instance_image_vis.clamp(0, 1)

        # Save the synthesized image
        instance_image_path = f"instance_image_{index}.png"
        TF.to_pil_image(instance_image_vis).save(instance_image_path)

        # Save each mask
        for i, mask in enumerate(instance_masks):
            mask_vis = mask.squeeze() * 255
            mask_path = f"instance_mask_{index}_mask_{i}.png"
            TF.to_pil_image(mask_vis.byte()).save(mask_path)
        #-------------------------------------------------------------------#

    def visualize_synthsized_imgs(self, synth_image, synth_masks, index):
        print(f"synth_image shape: {synth_image.shape}")
        #----- Save the synthesized image and masks for visualization -----#
        # Unnormalize the image tensor if it's normalized to [-1, 1]
        synth_image_vis = (synth_image + 1) / 2
        synth_image_vis = synth_image_vis.clamp(0, 1)

        # Save the synthesized image
        synth_image_path = f"synth_image_{index}.png"
        TF.to_pil_image(synth_image_vis).save(synth_image_path)

        # Save each mask
        for i, mask in enumerate(synth_masks):
            mask_vis = mask.squeeze() * 255
            mask_path = f"synth_mask_{index}_mask_{i}.png"
            TF.to_pil_image(mask_vis.byte()).save(mask_path)
        #-------------------------------------------------------------------#

    def collate_fn(self, examples):
        assert len(examples) == 1 
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        masks = [example["instance_masks"] for example in examples]
        token_ids = [example["token_ids"] for example in examples]

        input_ids = [example["synth_prompt_ids"] for example in examples] + input_ids
        pixel_values = [example["synth_image"] for example in examples] + pixel_values
        synth_masks = [example["synth_masks"] for example in examples]
        synth_token_ids = [example["synth_token_ids"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        # print(f"Pixel values shape: {pixel_values.shape}")
        # save_image(pixel_values, 'pixel_values_grid.png', nrow=8)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)
        masks = torch.stack(masks)
        synth_masks = torch.stack(synth_masks)
        token_ids = torch.stack(token_ids)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "instance_masks": masks,
            "synth_masks": synth_masks,
            "token_ids": token_ids,
            "synth_token_ids": synth_token_ids,
        }

        is_prior_flags = torch.tensor([ex["is_prior"] for ex in examples])
        batch["is_prior"] = is_prior_flags

        return batch

    def __getitem__(self, index):
        '''
        Generate 1 uniform sampled image from a training entry and 1 synthesized image across training entries
        '''
        if self.use_bg_tokens:
            bg_token_idx = random.randrange(len(self.bg_placeholder_tokens))
        else:
            bg_token_idx = -1

        ###------------------- Sampling concept combinations from the original image -------------------###
        if self.use_all_sythn:
            img_id_map, mask_ids, tokens_ids_to_use_global, prompt = self.generate_concept_comb_across_train_imgs()
            instance_image, instance_masks = self.synthesize_across_train_imgs(img_id_map, mask_ids)
        else:
            instance_image, instance_masks, tokens_ids_to_use_global, prompt = self.uniform_sample_train_img(index, bg_token_idx)

        # --- randomly swap instance image with prior ---
        is_prior = False
        if self.use_prior_data and random.random() < self.prior_prob:
            prior_idx = random.randrange(len(self.prior_imgs))
            instance_image = self.prior_imgs[prior_idx]
            instance_masks = torch.zeros_like(instance_masks)  # dummy mask
            tokens_ids_to_use_global = []
            prompt = f"a photo of a {self.subject_name}, on a simple white background"
            is_prior = True

        example = {
            "instance_images": instance_image,
            "instance_masks": instance_masks,
            "token_ids": torch.tensor(tokens_ids_to_use_global),
            "instance_prompt_ids": self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids,
            "is_prior": is_prior,
        }

        ###------------------- Sampling concept combinations across images -------------------###
        if self.use_all_instance:
            ### Note: this is used to run BaS baseline
            synth_image, synth_masks, synth_tokens_ids_to_use_global, synth_prompt = self.uniform_sample_train_img(index, bg_token_idx)
        else:
            ### First sample the concept combination
            if self.synth_type == 'random-no-overlap':
                img_id_map, synth_mask_ids, synth_tokens_to_use = self.generate_concept_comb_across_train_imgs()
            else:
                img_id_map, synth_mask_ids, synth_tokens_ids_to_use_global, synth_prompt= self.generate_concept_comb_across_train_imgs(bg_token_idx)
            
            ### Syth image using the sampled the concept combination
            if self.synth_type == '4-corner':
                synth_image, synth_masks = self.synthesize_across_train_imgs_4_corner(img_id_map, synth_mask_ids)
            elif self.synth_type == '2-subject':
                synth_image, synth_masks = self.synthesize_across_train_imgs_2_subject(img_id_map, synth_mask_ids)
            elif self.synth_type == 'random-no-overlap':
                synth_image, synth_masks, placed_img_id_map, placed_synth_mask_ids, placed_synth_tokens = self.synthesize_across_train_imgs_random_no_overlap(img_id_map, synth_mask_ids, synth_tokens_to_use)
                img_id_map, synth_mask_ids, synth_tokens_ids_to_use_global, synth_prompt = self.rectify_sampled_concepts(placed_img_id_map, placed_synth_mask_ids, placed_synth_tokens)
            elif self.synth_type == 'random-overlap':
                synth_image, synth_masks = self.synthesize_across_train_imgs_random_overlap(img_id_map, synth_mask_ids, bg_token_idx)

        example["synth_image"] = synth_image
        example["synth_masks"] = synth_masks
        example["synth_token_ids"] = torch.tensor(synth_tokens_ids_to_use_global)
        example["synth_prompt_ids"] = self.tokenizer(
            synth_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])
            example["synth_image"] = TF.hflip(example["synth_image"])
            example["synth_masks"] = TF.hflip(example["synth_masks"])

        return example