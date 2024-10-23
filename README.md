[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center"><img src="./docs/images/Blueprints-logo.png" width="300" alt="blueprints_logo"/></p>

# Federated LLM Fine-tuning with Private Data

This blueprint guides you through getting started with federated fine-tuning of LLMs. It utilizes Flower for Federated Learning and Differential Privacy and Hugging Face’s PEFT (Parameter-Efficient Fine-Tuning) for Fine-Tuning the LLM.

<p align="center"><img src="https://www.dailydoseofds.com/content/images/size/w600/format/webp/2023/11/federated-gif.gif"  alt="fl_diagram"/></p>


_(Figure source [link](https://www.dailydoseofds.com/federated-learning-a-critical-step-towards-privacy-preserving-machine-learning/))_

## Table of Contents

* [Project overview](#project-overview)
  * [Technical specifications](#technical-specifications)
  * [Federated-Learning](#federated-learning)
  * [Parameter-efficient Fine-tuning](#parameter-efficient-fine-tuning)
  * [Differential Privacy](#differential-privacy)
* [Getting Started](#getting-started)
* [Adapt to your needs](#adapt-to-your-needs)
* [Limitations](#limitations)
* [Future features](#future-features)
* [Resources](#resources)
* [Contribution Guidelines](#contribution-guidelines)
* [License](#license)


## Project Overview

This Blueprint aims to provide a foundation for a robust and privacy-preserving solution for distributed AI, ensuring that private data remains on-device while collaborating on model improvement. 

### Technical specifications

Built with:
- [Python](https://www.python.org/) (3.10 or higher)
- [Torch](https://pytorch.org/) + [TRL](https://github.com/huggingface/trl) for Supervised Fine-Tuning
- [Flower](https://github.com/adap/flower) for Federated Learning & Differential Privacy
- [HF's PEFT](https://github.com/huggingface/peft) for Parameter Efficient Fine-Tuning
- _(Optional)_ [Pre-commit](https://github.com/pre-commit/pre-commit) + [Black](https://black.readthedocs.io/en/stable/) + [Isort](https://github.com/PyCQA/isort) for consistent code formatting

#### Server requirements:
- 12B RAM for [OpenLLaMA 3Bv2](https://huggingface.co/openlm-research/open_llama_3b_v2) with 4-bit quantization

#### Client requirements:
- 2GB RAM for [OpenLLaMA 3Bv2](https://huggingface.co/openlm-research/open_llama_3b_v2) with 4-bit quantization

### Why Federated Learning?

Federated Learning (FL) is a collaborative approach to training machine learning models where data remains decentralized. Instead of bringing data to the model, federated learning brings the model to the data. Devices like smartphones or laptops train the model locally on their own data and then share only the model updates (gradients) with a central server, which aggregates them to improve the global model. This method enhances privacy by ensuring that raw data never leaves the local devices, reducing the risk of data breaches and complying with data protection regulations.

### Why Parameter-efficient Fine-tuning?

One challenge of FL is the need to transfer models to remote devices for computation, and Large Language Models (LLMs) can be substantial in size (often several GBs), which can pose significant storage and bandwidth issues. By utilizing Parameter-Efficient Fine-Tuning (PEFT), we dramatically reduce these demands during training. PEFT focuses on updating only a small subset of the model's parameters, minimizing data transfer and computational load. This approach optimizes the adaptation of pre-trained models to new tasks by selectively fine-tuning a fraction of the model’s parameters or adding a limited number of additional parameters. This significantly lowers the resources required for fine-tuning, making it feasible to leverage large models on devices with limited capabilities, without the need for extensive hardware.

### Why Differential Privacy?

Another challenge of FL is that, even though raw data is not transmitted, model updates can potentially leak information about local data if not properly protected. Differential privacy provides formal privacy guarantees by introducing controlled noise into data or computations. This ensures that the inclusion or exclusion of a single data point does not significantly affect the outcome of a query, thereby protecting individual privacy within the dataset. Even if someone knows every other data point in the dataset, they cannot infer the presence or absence of a specific individual's data. Differential privacy is crucial in federated learning environments as it enhances the privacy of aggregated model updates, ensuring that sensitive information about individuals cannot be reverse-engineered from shared data.

## Getting Started


Follow these steps to set up and run the project:

1. Clone the repository

```bash   
git clone https://github.com/mozilla-ai/Blueprint-local-federated-llm-fine-tuning.git
cd Blueprint-local-federated-llm-fine-tuning
```

2. Create a virtual environment and install the dependencies. 

**_Note:_** If you are going to develop the project further, it is highly recommended you install the `[dev]` version which includes pre-commit. Pre-commit will automatically execute [back](https://black.readthedocs.io/en/stable/) + [isort](https://github.com/PyCQA/isort) at every git commit to help you keep a consistent code format.


```bash
python -m venv ./venv
source venv/bin/activate
pip install -e .  # or pip install -e .[dev]
```

3. Run in _simulation mode_ with a 4-bit [OpenLLaMA 3Bv2](https://huggingface.co/openlm-research/open_llama_3b_v2) model. By default, there are 20 clients and there is a 10% strategy fit per round for 100 FL server rounds.

```bash
flwr run . --run-config "model.name='openlm-research/open_llama_3b_v2' model.quantization=4"
```

4. Evaluate the fine-tuned LLM with your own user input.


```bash
python evaluate.py --peft-path=/path/to/trained-model-dir/ \
    --question="What are some symptoms of covid-19?"
```


## Adapt to your needs

1. Add your own data to the repo and modify `data.py` if necessary.

2. Configure training arguments, server and client by modifying `pyproject.toml`. For more details about the configuration follow Flower's [documentation](https://flower.ai/docs/framework/how-to-configure-clients.html).

3. Run either in [simulation](https://flower.ai/docs/framework/how-to-run-simulations.html) mode or [deployment](https://flower.ai/docs/framework/explanation-flower-architecture.html) mode.

## Limitations

- When training on remote clients, the participating devices can vary widely in terms of hardware capabilities, network connectivity, and data distributions. This heterogeneity can lead to non-uniform training progress, complicating the aggregation and synchronization of model updates.

## Future features

Some great additions to this project could be the following:

- **Containerize the process**: Use Docker (and K8s?) to further improve mobility and deployment across machines.

- **Homomorphic Encryption**: A form of encryption that allows computations to be performed on encrypted texts, generating a result that when decrypted, matches the result of operations performed on the plaintext. This means data can be processed without ever being decrypted, protecting its privacy.

- **Activation Compression**: This technique reduces the size of the intermediate activations (outputs) generated during neural network computations, decreasing the amount of data transmitted between layers or devices, which helps in optimizing bandwidth usage and computational load.

## Resources

To get a better understanding of the concepts above, feel free to check out the following resources:

- [Official implementation from Flower with PEFT for Fine-tuning LLMs](https://github.com/adap/flower/tree/main/examples/flowertune-llm)
- [Official implementation from Flower with HF's Transformers](https://flower.ai/docs/framework/tutorial-quickstart-huggingface.html)
- [A short blog post on Federated Learning with helpful figures](https://www.dailydoseofds.com/federated-learning-a-critical-step-towards-privacy-preserving-machine-learning/)
- [DeepLearning.AI's course on Federated Learning](https://learn.deeplearning.ai/courses/intro-to-federated-learning/lesson/1/introduction)
- [DeepLearning.AI's course on Federated Fine-tuning of LLMs with Private Data](https://learn.deeplearning.ai/courses/intro-to-federated-learning-c2/lesson/1/introduction)

## Contribution Guidelines

We welcome contributions from the community! If you have ideas, suggestions, or improvements, feel free to open an issue or submit a pull request.
For more details on Mozilla's code of conduct and etiquette guidelines, please read the [Mozilla Community Participation Guidelines](https://www.mozilla.org/about/governance/policies/participation/). 


## License

This project is licensed under the Apache 2.0 License - see the `/docs/LICENSE` file for details.