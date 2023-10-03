import os
import hydra
import torch
import numpy as np
from src.data.split import Splitter
from src.data_fixes.infer import Inferrer
from src.data.concept_loader import ConceptLoader

from src.downstream_tasks.outcomes import OutcomeMaker


@hydra.main(config_path="../../configs/data", config_name="pretrain")
def main(cfg):
    # Load concepts and patients_info
    concepts_plus, patients_info = ConceptLoader()(
        concepts=["diagnose", "medication", "custom_events", "labtests"],
        data_dir=cfg.loader.data_dir,
        patients_info=cfg.loader.patients_info,
    )

    # Get the set of relevant patients
    pids = torch.load(os.path.join(cfg.paths.extra_dir, "PIDs.pt"))
    excluder_kept_indices = torch.load(
        os.path.join(cfg.paths.extra_dir, "excluder_kept_indices.pt")
    )
    patient_set = [pids[i] for i in excluder_kept_indices]

    # Filter out irrelevant patients (due to _plus element)
    concepts_plus = concepts_plus[concepts_plus.PID.isin(patient_set)]

    # Infer missing values
    concepts_plus = Inferrer()(concepts_plus)

    # Create outcomes
    outcomes = OutcomeMaker(cfg)(concepts_plus, patients_info, patient_set=patient_set)

    # Save outcomes
    torch.save(outcomes, os.path.join(cfg.paths.data_dir, "outcomes.pt"))

    # Split outcomes
    outcome_splits = Splitter(
        cfg,
        split_name="covid_splits.pt",
    )(outcomes, mode="covid")

    # Save features
    for set, set_outcomes in outcome_splits.items():
        torch.save(
            set_outcomes,
            os.path.join(cfg.paths.data_dir, f"{set}_{cfg.paths.outcomes_suffix}.pt"),
        )


if __name__ == "__main__":
    main()
