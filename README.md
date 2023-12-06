# Applied Differential Privacy for LLMs

## Team: Privacy Veil
- Pals Chinnakannan
- Jenn Yonemitsu
- Madhukar Yedulapuram
- Francisco Laplace

## Introduction

The purpose of this repository is to showcase the vulnerabilities and limitations of LLM fine-tuning and prompt tuning methods in preserving user privacy. By providing various proof of concepts and code examples, we aim to raise awareness about the potential risks associated with these techniques.

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

### Setting Up the Deployment Environment

1. **Login to the EC2 Instance:**
    - Use SSH to log in to your EC2 instance and create a deployment folder structure. For example:

    ```bash
    mkdir -p llm/meta-llama/deployment
    cd llm/meta-llama/deployment
    ```

2. **Create a Python Virtual Environment:**

    ```bash
    python3 -m venv llama-deploy
    source llama-deploy/bin/activate
    ```

3. **Clone the Repository:**

    - Clone the git repository in the deployment directory and install the required packages:

    ```bash
    git clone https://github.com/pals-ucb/privacy-veil.git
    pip install -r privacy-veil/requirements.txt
    ```

4. **Prepare the Trained Models:**

    - Copy the trained models into the `privacy-veil/instance/models/` folder. Refer to the companion training document for details on training the models.

### Configuring the Environment

Set up the necessary environment variables:

- **PV_DEVICE**:
    - This variable specifies the device that PyTorch will use to push the models. It's not limited to "cuda" or "cpu"; for instance, "mps" is supported for MAC M1|M2|M3.

    ```bash
    export PV_DEVICE="cuda" # or "cpu" or "mps"
    ```

- **PV_MODEL_PATH**:
    - Set this variable to the full path of the models directory, or a path relative to the instance folder.

    ```bash
    export PV_MODEL_PATH="/path/to/privacy-veil/instance/models"
    # or for a relative path:
    export PV_MODEL_PATH="./models"
    ```

### Launching the App

Run the Flask application with the following command:

```bash
flask --app pv-app run --host "0.0.0.0" --port 8080 --debug
```

## License

This project is licensed under the [Apache License](LICENSE).
