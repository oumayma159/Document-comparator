# Document-comparator

This project compares existing PDF files with new documents.

## Key Features

* Organize the files.
* Compare files based on semantics.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The things you need before installing the software.

* [Python](https://www.python.org/)
* [pip](https://pip.pypa.io/en/stable/installation/)

### Setup

A step by step guide that will tell you how to get the development environment up and running.

1. Clone the repository.

    ```sh
   git clone https://github.com/oumayma159/Document-comparator.git
    ```

1. Create a virtual environment.

    ```sh
    python -m venv .venv
    ```

1. Activate the virtual environment.

    ```sh
    # On Windows
    ./.venv/Scripts/activate
    # On Linux
    source ./.venv/bin/activate
    ```

1. Install the requirements.

    ```sh
    python -m pip install -r requirements.txt
    ```

    For GPU acceleration, you can visite PyTorch’s [Start Locally](https://pytorch.org/get-started/locally/) page.

## Usage

1. Add some documents to `data/base_documents`.
1. Compare a file to the existing documents.

    ```sh
    python main.py <file>
    ```

Note: you can use some PDF files from [tpn/pdfs](https://github.com/tpn/pdfs) to test the project.
