import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of <asset0> at the beach"
    )
    parser.add_argument("--prompts", 
                        nargs='+',
                        default=None,
                        help="Multiple inference prompts.",
    )
    parser.add_argument("--output_path", type=str, default="outputs/result.jpg")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()