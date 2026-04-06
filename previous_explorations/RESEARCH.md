# iEEG-to-scalp EEG denoising: a novel idea with fundamental barriers

**The proposed pipeline — using intracranial EEG forward-projected to the scalp as "ground truth" for training a deep learning denoiser — is genuinely novel but faces three near-fatal technical barriers that severely limit its viability as either a publication or a startup.** Partial brain coverage from iEEG electrodes (only 5–15% of cortex), severe data scarcity (~20–30 open-access hours of simultaneous recordings), and systematic forward-model biases combine to make the full pipeline impractical for general-purpose EEG denoising. Commercially, the February 2026 release of Zyphra's ZUNA — a 380M-parameter open-source EEG foundation model trained on 2 million channel-hours — eliminates any market opportunity for a standalone EEG cleaning tool. The core insight about the "ground truth problem" in EEG denoising is valid and publishable, but the execution path needs fundamental redesign.

---

## The forward model works; the real problem is what you're projecting

The theoretical foundation of the pipeline is actually its strongest element. The forward problem — projecting known source signals to the scalp via a lead field matrix — is mathematically well-posed and deterministic, unlike the notoriously ill-conditioned inverse problem. Given known iEEG electrode locations, source signals, and a realistic head model, scalp potentials can be computed as **V_scalp = L × J**.

However, empirical validation reveals that forward model accuracy is limited by several factors. The **Localize-MI dataset** (Mikulan et al., 2020), which validated electrical source imaging using simultaneous HD-EEG and intracranial stimulation in epilepsy patients, found mean localization errors of **5.39 mm (SD 2.61 mm)** under optimal parameter selection, with a max of 12.16 mm. Skull conductivity is the dominant error source: the brain-to-skull conductivity ratio estimates in the literature range from **15:1 to 80:1**, and mis-estimating this ratio can produce localization errors up to 31 mm and amplitude errors of **20–40%**. A recent empirical validation by Piastra et al. (2024) compared FEM-simulated potentials against actual sEEG measurements and concluded that "commonly used strategies to validate volume conduction models based solely on simulations might give an overly optimistic idea about volume conduction model accuracy."

These errors alone are not necessarily fatal for training a denoiser — deep learning can tolerate noisy targets. **The real showstopper is partial brain coverage.** iEEG electrodes in epilepsy patients are implanted to map seizure foci, typically covering only **5–15% of cortical surface area**, concentrated in temporal and frontal lobes. Forward-projecting these signals would produce a scalp EEG pattern that represents a small fraction of true cortical activity — missing posterior alpha rhythms, occipital visual responses, parietal contributions, and contralateral hemisphere signals. A denoiser trained on these targets would learn to reconstruct a fundamentally incomplete version of brain activity, effectively teaching the network that "clean" EEG contains only temporal-lobe contributions.

## Only ~20 hours of usable open-access data exist worldwide

The data landscape is starkly insufficient for training a deep learning model. Simultaneous iEEG and scalp EEG recordings are extremely rare because they require invasive electrode implantation alongside research-grade scalp recording, and they exist only in epilepsy monitoring contexts.

| Dataset | Year | Subjects | Modalities | Est. duration | Access |
|---------|------|----------|-----------|---------------|--------|
| Zurich Working Memory (Boran/Dimakopoulos) | 2020/2022 | 15 | Scalp EEG + iEEG + single units | ~8–12 hrs | Open (OpenNeuro ds004752) |
| Localize-MI (Mikulan et al.) | 2020 | 10–20 | HD-EEG (256ch) + sEEG stimulation | ~2–5 hrs | Open |
| Milan SPES Extended (Russo et al.) | 2022 | 36 | HD-EEG (256ch) + sEEG (192ch) | ~5–10 hrs | Open |
| Geneva Hidden IEDs (De Stefano et al.) | 2024 | 8 | HD-EEG + sEEG (interictal) | ~2–4 hrs | Open |
| Oxford iEEG-MEG (Zhang et al.) | 2021 | 11 | MEG (306ch) + iEEG | ~1–2 hrs | Restricted |

**Total open-access simultaneous iEEG + scalp EEG: approximately 17–31 hours.** This is 2–3 orders of magnitude below what deep learning models typically require. Non-open clinical monitoring data at centers in Bucharest, Mount Sinai, Nancy, and Shanghai may total 600–1,800 hours but remains locked behind institutional and ethical barriers. Critically, most open datasets contain **evoked/stimulation responses** (SPES protocols), not the spontaneous brain activity needed to train a general-purpose denoiser.

The iEEG-MEG bridge pathway faces even worse data scarcity: only about **4–6 total hours** of simultaneous iEEG-MEG recordings exist across all known datasets. Large iEEG-only repositories (MNI Open iEEG Atlas with 106 subjects, RAM project with 251 subjects, iEEG.org) could provide source signal data for forward modeling, but they lack paired scalp recordings for validation.

## Current denoisers all share the same ground truth weakness — but ZUNA may have solved it differently

Every published deep learning EEG denoiser uses one of two compromised ground-truth strategies. The dominant approach is **semi-simulation**: take ICA-processed "clean" EEG segments, add synthetic artifacts (EOG, EMG) at various SNR levels, and train on these pairs. The EEGDenoiseNet benchmark (Zhang et al., 2021) codified this approach, and models like **EEGDnet** (transformer-based, 2022), **EEGDfus** (conditional diffusion, 2024, currently the top performer with correlation coefficients up to 0.992), and **DHCT-GAN** (2025) all follow this paradigm. The second approach uses ICA/ICLabel to create raw-vs-cleaned pairs directly, as in **IC-U-Net** (2022) and **ART** (2025). Both strategies suffer from circular dependency: the "clean" target is only as good as ICA, and ICA itself is imperfect.

The proposed approach correctly identifies this fundamental weakness. **No prior work has used iEEG forward-projected to scalp space as ground truth for training a deep learning denoiser.** The closest prior works are:

- **Sanei/Abdi-Sargezeh group** (2018–2025): Maps scalp → iEEG (the reverse direction) using GANs for IED detection — not denoising
- **DeepSIF** (Bin He group, 2022, PNAS): Uses forward-modeled synthetic EEG as training *input* for source imaging — same concept but different application
- **GEDAI** (2025 preprint): Uses the EEG lead field to define a "brain signal subspace" for classical (non-DL) denoising — conceptually adjacent but not data-driven

However, Zyphra's **ZUNA** (released February 2026) may have leapfrogged the ground-truth problem entirely. ZUNA is a **380M-parameter masked diffusion autoencoder** trained on approximately **2 million channel-hours from 208 public datasets** using a self-supervised masked channel reconstruction objective. By randomly dropping 90% of channels during training and forcing reconstruction from the remaining 10%, ZUNA learns deep cross-channel correlations without needing any explicit "clean" ground truth. It operates on arbitrary channel layouts using 4D rotary positional encoding (3D space + time) and significantly outperforms spherical-spline interpolation across multiple benchmarks. Released under Apache 2.0 with an MNE-compatible pip package, it is freely available to all researchers.

## The novelty is real but narrow — and the execution path needs rethinking

The approach scores well on novelty: **no one has attempted to use physics-grounded iEEG-derived signals as supervised targets for training an EEG denoiser.** This correctly identifies the most fundamental limitation in the field — that ground truth is circular. The insight alone could support a workshop paper or perspective piece.

However, four technical risks would need to be addressed for a credible full publication:

- **Partial coverage bias**: The reconstructed "clean" EEG would be structurally incomplete, missing signals from ~85–95% of the cortex. A denoiser trained on this target would learn a distorted prior about what clean EEG looks like.
- **Systematic forward model errors**: Unlike random noise (which DL handles well), forward model biases from skull conductivity uncertainty are deterministic and consistent, potentially teaching the network systematically distorted spatial patterns — **20–40% amplitude errors** and topographic distortions baked into every training example.
- **Epilepsy domain gap**: All iEEG data comes from drug-resistant epilepsy patients with interictal discharges, background slowing, medication effects, and often structural lesions. A denoiser trained on this population may fail on healthy subjects.
- **iEEG-to-dipole conversion**: Translating iEEG local field potentials into equivalent current dipole moments for forward projection requires assumptions about source extent and orientation that introduce additional uncertainty.

A more viable reformulation might focus on **regional denoising** (only for brain areas covered by iEEG), use the forward model as a **data augmentation strategy** rather than sole ground truth, or pursue the **paired-data paradigm** directly — using simultaneous scalp-iEEG recordings without the forward modeling step.

## The commercial window has already closed

The market for EEG signal processing tools spans research (~$200–400M), clinical EEG (~$600–800M), and consumer BCI ($500M+), with the broader BCI market growing at **14–18% CAGR** toward $8–13B by 2033. Neurotech funding has exploded from $663M in 2022 to an estimated **$4.8B in 2025**.

But a standalone EEG cleaning tool is commercially unviable for several reinforcing reasons:

- **ZUNA is free**: A well-funded AI lab has released a state-of-the-art EEG foundation model under Apache 2.0, with MNE integration. It performs denoising, reconstruction, and upsampling across arbitrary channel layouts. Any paid cleaning product must dramatically outperform this.
- **Open-source dominance**: Research EEG runs on EEGLAB (MATLAB) and MNE-Python, both free. Automated pipelines like RELAX (2023) and ICLabel provide push-button artifact rejection at zero cost. Researchers are grant-funded and extremely price-sensitive.
- **No exit precedent**: No pure-play EEG signal processing company has achieved a significant exit. Acquisitions in neurotech have targeted hardware+software platforms (Meta acquired CTRL-Labs for ~$500M–1B) or complete clinical solutions (Ceribell IPO'd at $207M for rapid EEG seizure detection), not standalone preprocessing tools.
- **Low defensibility**: ML models for EEG denoising are published frequently (dozens of papers in 2024–2025 alone), architectures are open, and the EEGDenoiseNet benchmark enables anyone to train competitive models.

The existing commercial competitors — Intheon/Syntrogi (NeuroPype cloud API), BESA (€4K–12K desktop licenses), and BrainAccess (hardware + cloud AI) — further illustrate the challenge. Intheon already offers cloud-based real-time artifact rejection via API. BESA has decades of brand equity in academic source analysis. Neither has achieved a large exit.

If pursuing this commercially, the only viable paths would be: (1) **B2B SDK licensing to BCI OEMs** ($50–200K/year, targeting companies like Neurosity, Neurable, or Emotiv who need embedded signal quality), (2) **clinical validation with FDA/CE clearance** for specific use cases like automated epilepsy monitoring artifact rejection (a 2–3 year regulatory path that creates a real moat), or (3) **pharma/CRO services** for CNS drug trials where clean EEG biomarkers justify premium pricing.

## Key risks and realistic next steps

| Risk | Severity | Mitigation |
|------|----------|------------|
| Partial iEEG brain coverage | **Fatal** for general denoising | Restrict to regional denoising or use as augmentation only |
| Data scarcity (~20-30 hrs open-access) | **Severe** | Collaborate with 2-3 epilepsy centers; use synthetic augmentation |
| Forward model systematic biases | **High** | Individual FEM head models with calibrated conductivities |
| Epilepsy patient domain gap | **High** | Interictal-only epochs; validate on healthy EEG downstream tasks |
| ZUNA/open-source competition | **Fatal** for commercial standalone | Integrate as feature in broader platform |

**For a publication**: The core novelty — physics-grounded ground truth for EEG denoising — is publishable as a proof-of-concept, but scope must be narrowed. Focus on a specific cortical region (e.g., temporal lobe in the Zurich dataset), compare against ICA-based ground truth, and honestly characterize limitations. Target a methods-focused venue (Journal of Neural Engineering, NeuroImage). Confidence in publishability: **moderate** (6/10) if scoped correctly, but results are unlikely to show SOTA on general benchmarks.

**For a startup**: Do not pursue as a standalone EEG cleaning company. The technical barriers (partial coverage, data scarcity) and commercial barriers (ZUNA is free, open-source competition, no exit precedent) make this unviable. If drawn to the EEG/BCI space, consider the iEEG-EEG ground truth insight as one element of a broader clinical EEG intelligence platform targeting a specific vertical (epilepsy monitoring, sleep staging, or pharma biomarkers) where regulatory clearance creates defensibility.
