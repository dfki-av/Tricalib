---
title: 'Tri-Calib: An Interactive Target-Free and Keypoint-Based Extrinsic Calibration Tool for Tri-Modal Sensor Suites'

tags:
  - Python
  - LiDAR
  - event camera
  - sensor calibration
  - robotics
  - autonomous driving
  - neuromorphic

authors:
  - name: Rahul Jakkamsetty
    corresponding: true
    orcid: 0009-0000-0711-229X
    affiliation: 1
  - name: Ramy Battrawy
    affiliation: 1
  - name: René Schuster
    affiliation: "1, 2"
  - name: Didier Stricker
    affiliation: "1, 2"

affiliations:
  - index: 1
    name: DFKI – German Research Center for Artificial Intelligence, Kaiserslautern, Germany
  - index: 2
    name: RPTU – University of Kaiserslautern-Landau, Kaiserslautern, Germany
date: 24 July 2026

bibliography: paper.bib

header-includes:
  - \usepackage{pifont}
  - \newcommand{\cmark}{\ding{51}}
  - \newcommand{\xmark}{\ding{55}}
---

# Summary
Autonomous robots and vehicles increasingly rely on sensor suites that combine LiDAR scanners, RGB cameras, and event cameras to build robust representations of the environment. Before the data from these sensors can be fused, each sensor's position and orientations relative to the others must be precisely measured. This process is known as *extrinsic calibration*. Most existing calibration tools either require physical calibration targets (checkerboards, LED arrays) placed in the scene or are limited to calibrating only two sensors at a time. Neither approach naturally accommodates event cameras, a relatively new sensor type that captures only changes in brightness rather than full images. 


`Tri-Calib` is a cross-platform, open-source desktop application that allows researchers and engineers to calibrate a LiDAR scanner, an RGB camera, and an event camera together in a single interactive session, without any calibration targets. The user loads a set of temporally aligned data, clicks on corresponding features visible in all three sensor views, and the software computes the geometric transformations needed to align the data. The result is exported as a JSON file ready for use in downstream processing pipelines.


# Statement of Need
The combination of LiDAR, RGB, and event cameras is appearing in robotics research platforms, neuromorphic perception systems, and autonomous driving datasets [@dsec]. Accurate extrinsic calibration across all three modalities is a prerequisite for any multimodal fusion algorithm, yet it remains a practical bottleneck: errors in calibration propagate directly into downstream tasks, including point cloud colorization [@colorreg], object detection [@objdet], and simultaneous localization and mapping [@slam]. 


Existing open-source tools do not collectively address this combination. Kalibr [@kalibr] and camera-IMU toolboxes require calibration targets and do not support LiDAR or event cameras. Automatic targetless methods such as the Koide toolbox [@koide2023] and EdgeCalib [@edgecalib2023] support LiDAR–camera calibration but not event cameras. LCE-Calib [@lcecalib] supports event cameras alongside LiDAR but requires a physical checkerboard and performs only pairwise calibration. iKalibr [@ikalibr2025] offers comprehensive spatiotemporal calibration but mandates an IMU and continuous sensor motion, which is incompatible with single-frame static calibration workflows.


As per the \autoref{tab:table1}, `Tri-Calib` fills this gap. To the best of our knowledge, it is the only open-source tool that jointly calibrates LiDAR, RGB, and event cameras in a target-free, single-frame, initialization-free setting via an interactive GUI. The target audience is robotics and computer vision researchers who assemble custom sensor rigs and need a practical, accessible calibration workflow without specialized hardware setups.


| Tool                             | Target-free | LiDAR | RGB | Event | Joint Opt. | GUI |
|----------------------------------|:-----------:|:-----:|:---:|:-----:|:----------:|:---:|
| Kalibr [@kalibr] | \xmark | \xmark | \cmark | \xmark | \xmark | \xmark |
| OpenCalib [@opencalib] | ~  | \cmark | \cmark | \xmark | \xmark | ~ |
| LCE-Calib [@lcecalib] | \xmark | \cmark | \cmark | \cmark | \xmark | \xmark |
| iKalibr [@ikalibr2025] | \cmark | \cmark | \cmark | \xmark | \cmark | \xmark |
| Koide toolbox [@koide2023] | \cmark | \cmark | \cmark | \xmark | \xmark | ~ |
| EdgeCalib [@edgecalib2023] | \cmark | \cmark | \cmark | \xmark | \xmark | \xmark |
| Bertogalli et al. [@targetalignall] | \xmark | \cmark | \cmark | \cmark | \xmark | \xmark |
| **Tri-Calib (ours)** | \cmark | \cmark | \cmark | \cmark | \cmark | \cmark |

\label{tab:table1} Feature comparison of calibration tools. \cmark = supported, \xmark = not supported, ~ = partial support.

The choice to build a new tool rather than extend an existing one is justified by the structural incompatibility of current approaches with event cameras. Automatic targetless methods rely on dense image features that event cameras do not produce; target-based methods require purpose-built hardware that most event camera setups do not include. `Tri-Calib` sidesteps both constraints by delegating feature selection to the user, who can identify visually salient correspondences across the heterogeneous sensor outputs without additional hardware.

# Software Design
`Tri-Calib` is developed using Python and launches a multi-window graphical user interface built with PyQt6 [@pyqt6]. Three concurrent windows display the RGB image, the event frame, and a 3D point cloud rendered with PyVista [@pyvista]. Each window accepts interactive point selections, and inter-process communication is handled through Python multiprocessing polled by Qt timers, keeping each view responsive.

The key design trade-off is manual versus automatic correspondence selection. Automatic feature matching across LiDAR, RGB, and event modalities remains an open research problem: the three sensors produce fundamentally different representations (dense color images, sparse brightness-change maps, and unordered 3D point clouds) for which no robust cross-modal feature detector currently exists. Manual selection sacrifices throughput but is universally applicable. It imposes no constraints on scene type, sensor configuration, or environmental conditions, enabling use in both indoor and outdoor environments without modifications.

The tool offers three calibration pathways: (1) Perspective-n-Point (PnP) [@pnp] for LiDAR-to-RGB and LiDAR-to-Event (minimum 4 correspondences); (2) essential matrix estimation for RGB-to-Event (minimum 6 correspondences); and (3) a unified joint optimization that minimizes reprojection error across all modality pairs simultaneously using Levenberg-Marquardt, requiring only 4 correspondences total. The required number of correspondences quoted above is a theoretical minimum. Our experiments showed that 6–9 correspondences across all pathways result in better calibration compared to theoritical minimum. The joint optimizer parametrizes all three transformations as unit quaternions and translation vectors (21 parameters) and is the recommended pathway for calibrating all three sensors in a single session. Camera intrinsic parameters are loaded from JSON files; an optional coordinate-convention toggle converts between ROS/ENU and OpenCV frame conventions automatically. Calibration results, selected correspondences, and full application state are serialized to JSON for reproducibility and iterative refinement across sessions. 

A robustness study using Monte Carlo simulation confirms that the tool is reproducible across operators: injecting independent Gaussian pixel noise ($\sigma = 1$-$5$ px) into correspondences over 100 trials per configuration shows that seven correspondences achieve a coefficient of variation of 7.7% for LiDAR-to-RGB and 21.5% for LiDAR-to-Event at typical noise levels, with all 900 runs converging successfully.

# Research Impact Statement
Despite the relatively recent release of `Tri-Calib`, it has achieved notable impact in research and industrial projects. The tool was used to calibrate an LiDAR-camera system mounted to a truck to obtain ground-truth relative poses. Based on these, it enabled the training of a neural network for automated calibration from arbitrary natural scenes. Later, the tool was extensively employed in the research project COPPER to calibrate a tri-modal sensor suite in indoor and outdoor scenes. This impact ultimately led to a scientific publication [@lirecnet].

# AI Usage Disclosure
Anthropic's Claude Sonnet 5 was used to generate the HTML code for the documentation page for Tri-Calib. Additionally, the same tool was used to paraphrase the sentences in the manuscript. The VSCode's AI smart actions were used to generate the commit messages. All technical content, experimental results, and design decisions are the original work of the authors. The Authors confirm that AI-assisted text in documentation and manuscript was reviewed, edited, and verified for accuracy prior to submission.

# Acknowledgements

This work was partially funded by the Federal Ministry of Research, Technology, and Space (BMFTR), Germany, under the project COPPER (16IW24009).

# References