from dataclasses import dataclass

import torch
from torch import Tensor

import chai_lab.ranking.clashes as clashes
import chai_lab.ranking.plddt as plddt
import chai_lab.ranking.ptm as ptm
import chai_lab.ranking.utils as rutils
from chai_lab.utils.typing import Bool, Float, Int, typecheck


@typecheck
@dataclass
class SampleRanking:
    """Sample Ranking Data

    token_chain_masks: a tensor of shape (..., c, n) containing a boolean mask
        for each chain in the input
    token_asyms: a tensor of shape (c,) containing the unique asym ids for
        each chain in the sample. The token asyms are sorted numerically.
    token_chain_masks: a tensor of shape (..., c, n) containing a mask
        for each chain in the sample. The order of the chains is the same
        as token_asyms.
    aggregate_score: a tensor of shape (...) containing the aggregate ranking
        score for the sample
    ptm_scores: see ptm.get_scores for a description of the ptm scores
    clash_scores: a dictionary of clash scores
    plddt_scores: see plddt.PLDDTScores for a description of the plddt scores
    """

    token_chain_masks: Bool[Tensor, "... c n"]
    token_asyms: Int[Tensor, "c"]
    aggregate_score: Float[Tensor, "..."]
    ptm_scores: ptm.PTMScores
    clash_scores: clashes.ClashScores
    plddt_scores: plddt.PLDDTScores


@typecheck
def rank(
    atom_coords: Float[Tensor, "... a 3"],
    atom_mask: Bool[Tensor, "... a"],
    atom_token_index: Int[Tensor, "... a"],
    token_exists_mask: Bool[Tensor, "... n"],
    token_asym_id: Int[Tensor, "... n"],
    token_entity_type: Int[Tensor, "... n"],
    token_valid_frames_mask: Bool[Tensor, "... n"],
    # lddt
    lddt_logits: Float[Tensor, "... a lddt_bins"],
    lddt_bin_centers: Float[Tensor, "lddt_bins"],
    # pae
    pae_logits: Float[Tensor, "... n n pae_bins"],
    pae_bin_centers: Float[Tensor, "pae_bins"],
    # clash
    clash_threshold: float = 1.1,
    max_clashes: int = 100,
    max_clash_ratio: float = 0.5,
) -> SampleRanking:
    """
    Compute ranking scores for a sample.
    In addition to the pTM/ipTM aggregate score, we also return chain
    and inter-chain level statistics for pTM and clashes.
    see documentation for SampleRanking for a complete description.
    """

    ptm_scores = ptm.get_scores(
        pae_logits=pae_logits,
        token_exists_mask=token_exists_mask,
        valid_frames_mask=token_valid_frames_mask,
        bin_centers=pae_bin_centers,
        token_asym_id=token_asym_id,
    )
    clash_scores = clashes.get_scores(
        atom_coords=atom_coords,
        atom_mask=atom_mask,
        atom_asym_id=torch.gather(
            token_asym_id,
            dim=-1,
            index=atom_token_index.long(),
        ),
        atom_entity_type=torch.gather(
            token_entity_type,
            dim=-1,
            index=atom_token_index.long(),
        ),
        max_clashes=max_clashes,
        max_clash_ratio=max_clash_ratio,
        clash_threshold=clash_threshold,
    )

    plddt_scores = plddt.get_scores(
        lddt_logits=lddt_logits,
        atom_mask=atom_mask,
        bin_centers=lddt_bin_centers,
        atom_asym_id=torch.gather(
            token_asym_id,
            dim=-1,
            index=atom_token_index.long(),
        ),
    )

    # aggregate score
    aggregate_score = (
        0.2 * ptm_scores.complex_ptm
        + 0.8 * ptm_scores.interface_ptm
        - 100 * clash_scores.has_clashes.float()
    )

    chain_masks, asyms = rutils.get_chain_masks_and_asyms(
        asym_id=token_asym_id,
        mask=token_exists_mask,
    )

    return SampleRanking(
        token_chain_masks=chain_masks,
        token_asyms=asyms,
        aggregate_score=aggregate_score,
        ptm_scores=ptm_scores,
        clash_scores=clash_scores,
        plddt_scores=plddt_scores,
    )
