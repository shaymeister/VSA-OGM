## VSA-OGM - Occupancy Grid Mapping in 2D

In this application of bio-inspired vector symbolic architectures, we employ a novel hyperdimensional occupancy grid mapping system with Shannon entropy. For the most in-depth exploration of our experiments and results, please take a look at [our paper](.).

*This work was supported under grant 5.21 from the University of Michigan's Automotive Research Center (ARC) and the U.S. Army's Ground Vehicle Systems Center (GVSC).* 

<img src="./assets/toy-sim.gif" width="300" height="300"/> <img src="./assets/vsa-toysim-crop.gif" width="370" height="300" />

---

### Directory Structure

- **dataloaders**: individual classes to lead different dataset formats
- **datasets**: the Intel map and single-agent toysim datasets
- **experiments**: script files to run the single and multi-agent experiments with the toysim and Intel datasets
- **notebooks**: tools to evaluate against EviLOG, run the ablation study, evaluate runtime performance, and perform Shannon-based entropy extraction
- **metrics.py**: functions to calculate TP and FP rates along with AUC
- **plotting.py**: functions to visualize statistical results
- **utilities.py**: functions to split training dataset into train and test sets

---

### Datasets

- Toy Sim (Single Agent): included in `datasets`
- Toy Sim (Fusion): download [Agent 1](https://gmuedu-my.sharepoint.com/:u:/g/personal/ssnyde9_gmu_edu/EVNScsJma1lMpQmTgLmBmBoBaVgLRgwrIcVRiWLAOtHiqA?e=GrE7eq) and [Agent 2](https://gmuedu-my.sharepoint.com/:u:/g/personal/ssnyde9_gmu_edu/ETE2c01yROlIkH3-gLSo7vsBIKKOt1S_fgdVfthFgEgW3Q?e=aEAXiM)
- Intel (Single Agent): included in `datasets`
- Intel (Fusion): generate with [this notebook](./notebooks/datasets/intel_map_fusion_data.ipynb)
- EviLOG: download according to [their repository](https://github.com/ika-rwth-aachen/EviLOG)

---

### Experiments

Both the Intel map and Toy sim single agent experiments can be recreated with `vsa_map_single.py` in the `experiments` directory.

Conversely, the fusion experiments can be duplicated with `vsa_map_fusion.py` in the same directory. The runtime processing and entropy based information extraction were done with Jupyter in the `notebooks` directory.

We trained EviLOG with the author's codebase. Our evaluation on their dataset was conducted with Jupyter in the `notebooks/datasets` directory.

---

### Authors and Contact Information

- **Shay Snyder****: [ssnyde9@gmu.edu](ssnyde9@gmu.edu)
- **Andrew Capodieci**: [acapodieci@neyarobotics.com](acapodieci@neyarobotics.com)
- **David Gorsich**: [david.j.gorsich.civ@army.mil](david.j.gorsich.civ@army.mil)
- **Maryam Parsa**: [mparsa@gmu.edu](mparsa@gmu.edu)

If you have any issues, questions, comments, or concerns about Highfrost, please reach out to the corresponding author (**). We will respond as soon as possible.

---

### Reference and Citation

If you find our work useful in your research endeavors, we would appreciate if you would consider citing [our paper](.):

```text
@misc{snyder2024braininspiredprobabilisticoccupancy,
      title={Brain Inspired Probabilistic Occupancy Grid Mapping with Hyperdimensional Computing}, 
      author={Shay Snyder and Andrew Capodieci and David Gorsich and Maryam Parsa},
      year={2024},
      eprint={2408.09066},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.09066}, 
}
```

