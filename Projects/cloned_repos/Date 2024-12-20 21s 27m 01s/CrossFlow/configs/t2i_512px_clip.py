import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model = Args(
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim=768,
    num_clip_token=77,
    gradient_checking=True, # for larger model
    cfg_indicator=0.15,
    textVAE = Args(
        num_blocks = 11,
        hidden_dim = 1024,
        hidden_token_length = 256,
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
    stage_configs = [
            Args(
                block_type = "TransformerBlock", 
                dim = 1024,  # channel
                hidden_dim = 2048,
                num_attention_heads = 16,
                num_blocks = 65,  # depth
                max_height = 16,
                max_width = 16,
                image_input_ratio = 1,
                input_feature_ratio = 4,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
            Args(
                block_type = "ConvNeXtBlock", 
                dim = 512, 
                hidden_dim = 1024, 
                kernel_size = 7, 
                num_blocks = 33,
                max_height = 32,
                max_width = 32,
                image_input_ratio = 1,
                input_feature_ratio = 2,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
            Args(
                block_type = "ConvNeXtBlock", 
                dim = 256, 
                hidden_dim = 512, 
                kernel_size = 7, 
                num_blocks = 33,
                max_height = 64,
                max_width = 64,
                image_input_ratio = 1,
                input_feature_ratio = 1,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
    ],
)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.z_shape = (4, 64, 64)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=1000000,
        batch_size=1024,
        mode='cond',
        log_interval=10,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.00001,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    global model
    config.nnet = d(
        name='dimr',
        model_args=model,
    )
    config.loss_coeffs = [1/4, 1/2, 1]
    
    config.dataset = d(
        name='JDB_demo_features',
        resolution=512,
        llm='clip',
        train_path='/data/qihao/dataset/JDB_demo_feature/',
        val_path='/data/qihao/dataset/coco_val_features/',
        cfg=False,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=30000,
        mini_batch_size=10,
        cfg=False,
        scale=7,
        path=''
    )

    return config
