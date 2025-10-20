import unsloth
from unsloth import FastLanguageModel

def load_model(args, continue_path=None):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = continue_path or args.model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit = not args.load_in_8bit,
        load_in_8bit = args.load_in_8bit,
    )

    model = FastLanguageModel.get_peft_model(
        model, 
        r = args.lora_r,
        target_modules = [
        "lm_head",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        use_gradient_checkpointing = "unsloth",
        bias = "none",
        use_rslora = True,
        random_state = args.seed,
    )

    return model , tokenizer

