from setuptools import setup

setup(
    name="ehr2vec",
    description="A setup script for the ehr2vec repo",
    packages=[
        "src",
        "src.common",
        "src.data",
        "src.data_fixes",
        "src.dataloader",
        "src.downstream_tasks",
        "src.embeddings",
        "src.evaluation",
        "src.model",
        "src.trainer",
        "src.tree",
        "src.behrt",
    ],
    install_requires=["torch", "pandas", "transformers", "hydra-core", "pytorch-pretrained-bert"],
)
