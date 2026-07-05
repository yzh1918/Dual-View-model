# Dual-View-model
This repository contains the implementation of the Dual-View Collaborative Learning Network (DCLN), a multi-view semi-supervised rumor detection framework that combines BERT-based textual semantics with GAT-based propagation structures for accurate rumor detection under extremely low-resource settings.

Repository Structure

 MIX-alpha/            # Code for Chinese dataset (CED / Weibo)
MIX-pheme-alpha/      # Code for English dataset (PHEME)

MIX-alpha: Processes the Chinese rumor detection dataset (CED). Contains data preprocessing, model training, and evaluation scripts for the Chinese track.

MIX-pheme-alpha: Processes the English rumor detection dataset (PHEME). Contains data preprocessing, model training, and evaluation scripts for the English track.
Datasets

This work uses two publicly available datasets. Please refer to the original papers for download instructions:

CED (Chinese Weibo Dataset)

Song, C., Yang, C., Chen, H., Tu, C., Liu, Z., & Sun, M. (2021). CED: Credible Early Detection of Social Media Rumors. IEEE Transactions on Knowledge and Data Engineering, 33(8), 3035–3047.

PHEME (English Twitter Dataset)

Zubiaga, A., Liakata, M., & Procter, R. (2016). Learning Reporting Dynamics during Breaking News for Rumour Detection in Social Media. arXiv preprint arXiv:1610.07363.
