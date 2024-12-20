# Genesis-backflip

A functional and streamlined codebase for locomotion training in Genesis, featuring:
- Effective domain randomizations and useful rewards
- Fundamental rewards for executing a complete backflip

We use the same domain randomizations as in this repository in both single and double backflips. During deployment, we slightly changed Kps used on the real robot to do a clean backflip.

## Usage

1. Install the modified version rsl_rl

    ```
    cd rsl_rl && pip install -e .
    ```

2. Train the Unitree Go2 for a backflip

    This codebase includes only the fundamental rewards required for learning a backflip. You can observe how Go2 learns to perform a full backflip by running:
    ```
    python train_backflip.py -e EXP_NAME
    ```

3. Evaluate existing checkpoints

    Run
    ```
    python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
    ```

4. Generate a deployable checkpoint

    To produce a deployable checkpoint comparable to the provided example in Genesis, additional rewards must be incorporated to regularize motion and minimize torque peaks.