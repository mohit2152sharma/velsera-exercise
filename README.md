# verlsera-assignment-exercise

The finetuning analysis can be found in the notebook `experiments/finetuning.ipynb`. I have added more components to the project to make it more modular and complete.

> [!WARNING]
> Given the time constraint, I wasn't able to test the code thorughly and may contain bugs. Although, it does highlight the style and essential components required for the project. Most of the setup and foundational code is already written.

## Project Structure

- **Fine-tuning**: Finetuning related code is in `src/velsera/finetuning/`. Each step of finetuning is modularized into separate classes allowing for easier flexibility

  - A separate class for prompts, which allows to test and compare different prompts for fine-tuning
  - A separate function for generating reports in markdown format
  - `main.py` supports running multiple models in loop

- **Preprocessing**: Preprocessing related code is in `src/velsera/preprocessing/`.

  - This module reads the files from the directory
  - Cleans those files, extracts the required fields
  - Combines them into a dataframe

- **Web API**: A FastAPI-based web service in `src/velsera/webserver/` that:

  - Exposes the `classify_paper` endpoint for classifying the paper
  - Supports for loading the model only at the server startup time to avoid loading it at each request

> [!NOTE]
> It assumes the model is already downloaded and stored on the server, there isn't any code at the moment to download it from s3 or any other storage.

- **Experiments**: Jupyter notebook in `experiments/` demonstrating the model fine-tuning process and analysis.

  - File link: [here](experiments/finetuning.ipynb)
  - Details on model selection and accuracy improvements are included in the notebook.
  - TLDR: I selected the quantized llama model from `unsloth`. Given the time and resource constraint, it made sense to go with the smaller model. Before finetuning the accuracy was around 52% and after finetuning the accuracy jumped to 92%.

> [!NOTE]
> I didn't experiment with other models because of resource and time constraints.

- **Containerization**: Multi-stage Docker to optimize build time and image size:

- **Testing**: Setup for pytest and some basic tests return in `tests/`

- **CI/CD**: Github actions for CI/CD.

  - Runs the test on every push to the repository
  - Once the tests are successful, it builds and deploys the docker image to AWS ECR

> [!NOTE]
> The CI/CD pipeline is most likely to fail, as building images for GPU requires some special attention

- **Infra**: Infrastructure as a code for AWS resources is in the repo `infra/`

  - EKS cluster to deploy the inference model
  - S3 bucket to store the fine tuned model
  - ECR repository to store the docker image
  - Secrets manager to store the secrets
