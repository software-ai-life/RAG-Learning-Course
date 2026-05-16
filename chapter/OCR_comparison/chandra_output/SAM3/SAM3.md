![Figure 2: A 3x10 grid of images comparing SAM 3 and OWLv2 segmentation results. The columns are labeled with concepts: 'a white flower', 'young plant', 'hand towel', 'Adirondack chair', 'cheesecloth', 'dangle earring', 'rowboat', 'colander', and 'toilet roll holder'. The rows are labeled 'ORIGINAL', 'SAM 3', and 'OWLv2'. SAM 3 shows more precise and complete segmentation masks compared to OWLv2, which often misses parts of the objects or includes background noise.](3e3ab7edbd5fc448f282c6f9a6667170_1_img.webp)

**Figure 2** Examples of SAM 3 improving segmentation of open-vocabulary concepts compared to OWLv2 (Minderer et al., 2024), on the SA-Co benchmark. See §F.6.1 for additional SAM 3 outputs.

To fill this gap, we present SAM 3, a model that achieves a step change in promptable segmentation in images and videos, improving PVS relative to SAM 2 and setting a new standard for *Promptable Concept Segmentation (PCS)*. We formalize the PCS task (§2) as taking text and/or image exemplars as input, and predicting instance and semantic masks for every single object matching the concept, while preserving object identities across video frames (see Fig. 1). To focus on recognizing atomic visual concepts, we constrain text to simple noun phrases (NPs) such as “red apple” or “striped cat”. While SAM 3 is not designed for long referring expressions or queries requiring reasoning, we show that it can be straightforwardly combined with a Multimodal Large Language Model (MLLM) to handle more complex language prompts. Consistent with previous SAM versions, SAM 3 is fully interactive, allowing users to resolve ambiguities by adding refinement prompts to guide the model towards their intended output.

Our *model* (§3) consists of a detector and a tracker that share a vision encoder (Bolya et al., 2025). The detector is a DETR-based (Carion et al., 2020) model conditioned on text, geometry, and image exemplars. To address the challenge of open-vocabulary concept detection, we introduce a separate *presence head* to decouple recognition and localization, which is especially effective when training with challenging *negative phrases*. The tracker inherits the SAM 2 transformer encoder-decoder architecture, supporting video segmentation and interactive refinement. The decoupled design for detection and tracking avoids task conflict, as the detector needs to be identity agnostic, while the tracker’s main objective is to separate identities in the video.

To unlock major performance gains, we build a human- and model-in-the-loop *data engine* (§4) that annotates a large and diverse training dataset. We innovate upon prior data engines in three key ways: (i) *media curation*: we curate more diverse media domains than past approaches that rely on homogeneous web sources, (ii) *label curation*: we significantly increase label diversity and difficulty by leveraging an ontology and multimodal LLMs as “AI annotators” to generate noun phrases and hard negatives, (iii) *label verification*: we double annotation throughput by fine-tuning MLLMs to be effective “AI verifiers” that achieve near-human accuracy.

Starting from noisy media-phrase-mask pseudo-labels, our data engine checks mask quality and exhaustivity using both human and AI verifiers, filtering out correctly labeled examples and identifying challenging error cases. Human annotators then focus on fixing these errors by manually correcting masks. This enables us to annotate high-quality training data with 4M *unique* phrases and 52M masks, and a synthetic dataset with 38M phrases and 1.4B masks. We additionally create the Segment Anything with Concepts (SA-Co) *benchmark* for PCS (§5) containing 207K unique concepts with exhaustive masks in 120K images and 1.7K videos, > 50× more concepts than existing benchmarks.

Our *experiments* (§6) show that SAM 3 sets a new state-of-the-art in promptable segmentation, e.g., reaching a zero-shot mask AP of 48.8 on LVIS *vs.* the current best of 38.5, surpassing baselines on our new SA-Co benchmark by at least 2× (see examples in Fig. 2), and improving upon SAM 2 on visual prompts. Ablations (§A) verify that the choice of backbone, novel presence head, and adding hard negatives all boost results, and establish scaling laws on the PCS task for both our high-quality and synthetic datasets. We open-source the SA-Co benchmark and release the SAM 3 checkpoints and inference code. On an H200 GPU, SAM 3 runs in 30 ms for a single image with 100+ detected objects. In video, the inference latency scales with the number of objects, sustaining near real-time performance for ~ 5 concurrent objects. We review related work in §7; next, we dive into the task.![Figure 3: Illustration of supported initial and optional interactive refinement prompts in the PCS task. The figure shows four panels: 1. INITIAL PROMPT: An image of fish with a green bounding box around one fish. 2. OUTPUT: The same image with multiple purple masks around detected fish. 3. REFINEMENT PROMPTS: The same image with a green bounding box around one fish and a red dashed bounding box around another. 4. OUTPUT: The same image with refined purple masks.](6391bc6dbf0ba981e225b79350d978e3_1_img.webp)

**Figure 3** Illustration of supported initial and optional interactive refinement prompts in the PCS task.

## 2 Promptable Concept Segmentation (PCS)

We define the Promptable Concept Segmentation task as follows: given an image or short video ( $\leq 30$  secs), detect, segment and track all instances of a visual concept specified by a short text phrase, image exemplars, or a combination of both. We restrict concepts to those defined by simple noun phrases (NPs) consisting of a noun and optional modifiers. Noun-phrase prompts (when provided) are *global* to all frames of the image/video, while image exemplars can be provided on *individual* frames as positive or negative bounding boxes to iteratively *refine* the target masks (see Fig. 3).

All prompts must be consistent in their category definition, or the model’s behavior is undefined; e.g., “fish” cannot be refined with subsequent exemplar prompts of just the tail; instead the text prompt should be updated. Exemplar prompts are particularly useful when the model initially misses some instances, or when the concept is rare.

Our vocabulary includes any simple noun phrase groundable in a visual scene, which makes the task intrinsically ambiguous. There can be multiple interpretations of phrases arising from polysemy (“mouse” device *vs.* animal), subjective descriptors (“cozy”, “large”), vague or context-dependent phrases that may not even be groundable (“brand identity”), boundary ambiguity (whether ‘mirror’ includes the frame) and factors such as occlusion and blur that obscure the extent of the object. While similar issues appear in large closed-vocabulary corpora (e.g., LVIS (Gupta et al., 2019)), they are alleviated by carefully curating the vocabulary and setting a clear definition of all the classes of interest. We address the ambiguity problem by collecting test annotations from three experts, adapting the evaluation protocol to allow multiple valid interpretations (§E.3), designing the data pipeline/guidelines to minimize ambiguity in annotation, and an ambiguity module in the model (§C.2).

## 3 Model

SAM 3 is a generalization of SAM 2, supporting the new PCS task (§2) as well as the PVS task. It takes *concept* prompts (simple noun phrases, image exemplars) or *visual* prompts (points, boxes, masks) to define the *objects* to be (individually) segmented spatio-temporally. Image exemplars and visual prompts can be *iteratively* added on individual frames to *refine* the target masks—false positive and false negative objects can be *removed* or *added* respectively using image exemplars and an *individual* mask(let) can be refined using PVS in the style of SAM 2. Our architecture is broadly based on the SAM and (M)DETR (Carion et al., 2020; Kamath et al., 2021) series. Fig. 4 shows the SAM 3 architecture, consisting of a dual encoder-decoder transformer—a *detector* for image-level capabilities—which is used in combination with a *tracker* and memory for video. The detector and tracker ingest vision-language inputs from an aligned Perception Encoder (PE) backbone (Bolya et al., 2025). We present an overview below, see §C for details.

**Detector Architecture.** The architecture of the detector follows the general DETR paradigm. The image and text prompt are first encoded by PE and image exemplars, if present, are encoded by an exemplar encoder. We refer to the image exemplar tokens and text tokens jointly as “prompt tokens”. The fusion encoder then accepts the unconditioned embeddings from the image encoder and conditions them by cross-attending to the prompt tokens. The fusion is followed by a DETR-like decoder, where learned object queries cross-attend to the conditioned image embeddings from the fusion encoder.**Figure 4** SAM 3 architecture overview. See Fig. 10 for a more detailed diagram.

Each decoder layer predicts a classification logit for each object query (in our case, a binary label of whether the object corresponds to the prompt), and a delta from the bounding box predicted by the previous level, following Zhu et al. (2020). We use box-region-positional bias (Lin et al., 2023) to help focalize the attention on each object, but unlike recent DETR models, we stick to vanilla attention. During training, we adopt dual supervision from DAC-DETR (Hu et al., 2023), and the Align loss (Cai et al., 2024). The mask head is adapted from MaskFormer (Cheng et al., 2021). In addition, we also have a semantic segmentation head, which predicts a binary label for every pixel in the image, indicating whether or not it corresponds to the prompt. See §C for details.

**Presence Token.** It can be difficult for each of the proposal queries to both recognize (what) and localize (where) an object in the image/frame. For the recognition component, contextual cues from the entire image are important. However, forcing proposal queries to understand the global context can be counterproductive, as it conflicts with the inherently local nature of the localization objective. We decouple the recognition and localization steps by introducing a learned global *presence token*. This token is solely responsible for predicting whether the target concept in the form of a noun phrase (NP) is present in the image/frame, i.e.  $p(\text{NP is present in input})$ . Each proposal query  $q_i$  only needs to solve the localization problem  $p(q_i \text{ is a match} \mid \text{NP is present in input})$ . The final score for each proposal query is the product of its own score and the presence score.

**Image Exemplars and Interactivity.** SAM 3 supports image exemplars, given as a pair—a bounding box and an associated binary label (positive or negative)—which can be used in isolation or to supplement the text prompt. The model then detects all the instances that match the prompt. For example, given a positive bounding box on a dog, the model will detect *all* dogs in the image. This is different from the PVS task in SAM 1 and 2, where a visual prompt yields only a single object instance. Each image exemplar is encoded separately by the exemplar encoder using an embedding for the position, an embedding for the label, and ROI-pooled visual features, then concatenated and processed by a small transformer. The resulting prompt is concatenated to the text prompt to comprise the prompt tokens. Image exemplars can be *interactively* provided based on errors in current detections to refine the output.

**Tracker and Video Architecture.** Given a video and a prompt  $P$ , we use the detector and a tracker (see Fig. 4) to detect and track objects corresponding to the prompt throughout the video. On each frame, the detector finds new objects  $\mathcal{O}_t$  and the tracker propagates masklets  $\mathcal{M}_{t-1}$  (spatial-temporal masks) from frames at the previous time  $t - 1$  to their new locations  $\hat{\mathcal{M}}_t$  on the current frame at time  $t$ . We use a matching function to associate propagated masklets  $\hat{\mathcal{M}}_t$  with new object masks emerging in the current frame  $\mathcal{O}_t$ ,

$$\hat{\mathcal{M}}_t = \text{propagate}(\mathcal{M}_{t-1}), \quad \mathcal{O}_t = \text{detect}(I_t, P), \quad \mathcal{M}_t = \text{match\_and\_update}(\hat{\mathcal{M}}_t, \mathcal{O}_t).$$

**Tracking an Object with SAM 2 Style Propagation.** A masklet is initialized for every object detected on the first frame. Then, on each subsequent frame, the tracker module predicts the new masklet locations  $\hat{\mathcal{M}}_t$  of those already-tracked objects based on their previous locations  $\mathcal{M}_{t-1}$  through a single-frame propagation step similar to the video object segmentation task in SAM 2. The tracker shares the same image/frame encoder (PE backbone) as the detector. After training the detector, we freeze PE and train the tracker as in SAM 2, including a prompt encoder, mask decoder, memory encoder, and a memory bank that encodes the**Figure 5** Overview of the final SAM 3 data engine. See §E.1 for details of collected data.

object’s appearance using features from the past frames and conditioning frames (frames where the object is first detected or user-prompted). The memory encoder is a transformer with self-attention across visual features on the current frame and cross-attention from the visual features to the spatial memory features in the memory bank. We describe details of our video approach in §C.3.

During inference, we only retain frames where the object is confidently present in the memory bank. The mask decoder is a two-way transformer between the encoder hidden states and the output tokens. To handle ambiguity, we predict three output masks for every tracked object on each frame along with their confidence, and select the most confident output as the predicted mask on the current frame.

**Matching and Updating Based on Detections.** After obtaining the tracked masks  $\hat{\mathcal{M}}_t$ , we match them with the current frame detections  $\mathcal{O}_t$  through a simple IoU based *matching function* (§C.3) and add them to  $\mathcal{M}_t$  on the current frame. We further spawn new masklets for all newly detected objects that are not matched. The merging might suffer from ambiguities, especially in crowded scenes. We address this with two temporal disambiguation strategies outlined next.

First, we use temporal information in the form of a *masklet detection score* (§C.3) to measure how consistently a masklet is matched to a detection within a temporal window (based on the number of past frames where it was matched to a detection). If a masklet’s detection score falls below a threshold, we suppress it. Second, we use the detector outputs to resolve specific failure modes of the tracker due to occlusions or distractors. We periodically *re-prompt* the tracker with high-confidence *detection* masks  $\mathcal{O}_t$ , replacing the tracker’s own predictions  $\hat{\mathcal{M}}_t$ . This ensures that the memory bank has recent and reliable references (other than the tracker’s own predictions).

**Instance Refinement with Visual Prompts.** After obtaining the initial set of masks (or masklets), SAM 3 allows refining individual masks(lets) using positive and negative clicks. Specifically, given the user clicks, we apply the prompt encoder to encode them, and feed the encoded prompt into the mask decoder to predict an adjusted mask. In videos the mask is then propagated across the entire video to obtain a refined masklet.

**Training Stages.** We train SAM 3 in four stages that progressively add data and capabilities: 1) Perception Encoder (PE) pre-training, 2) detector pre-training, 3) detector fine-tuning, and 4) tracker training with a frozen backbone. See §C.4.1 for details.

## 4 Data Engine

Achieving a step change in PCS with SAM 3 requires training on a large, diverse set of concepts and visual domains, beyond existing datasets (see Fig. 12). We build an efficient data engine that iteratively generates annotated data via a feedback loop with SAM 3, human annotators, and *AI annotators*, actively mining media-phrase pairs on which the current version of SAM 3 fails to produce high-quality training data to further improve the model. By delegating certain tasks to AI annotators—models that match or surpass human accuracy—we more than double the throughput compared to a human-only annotation pipeline. We develop the data engine in four phases, with each phase increasing the use of AI models to steer human effort to the most challenging failure cases, alongside expanding visual domain coverage. Phases 1-3 focus only on images, with Phase 4 expanding to videos. We describe the key steps here; details and metrics are in §D.![Figure 6: Example video (top) and images (bottom) from SA-Co with annotated phrases and instance masks/IDs. The top row shows four video frames with instance masks and IDs. The bottom row shows six images with instance masks and IDs. Each image has a legend of colored boxes with text labels identifying the objects.](e59e7a26559ffe6c0333113844a966b7_1_img.webp)

Figure 6 displays example video (top) and images (bottom) from SA-Co with annotated phrases and instance masks/IDs. The top row shows four video frames with instance masks and IDs. The bottom row shows six images with instance masks and IDs. Each image has a legend of colored boxes with text labels identifying the objects.

**Top Row (Video Frames):**

- a tree
- the fabric
- a sheet of corrugated metal
- white sack
- a yellow shirt
- flip-flop
- tail light
- plaid sarong
- the white license plate
- a long-sleeved blue and white checkered shirt

**Bottom Row (Images):**

- white Persian cat
- a decorative trim
- gravel path
- a neatly trimmed bush
- the blue-green eye
- the red velvet chair
- the gold finial
- a dome-shaped roof
- the white metal frame
- the trellis
- a couch, daybed, a sheep, ...
- the large, yellow estate
- the small, white building
- plastic bag
- pomegranate
- a chain
- a cardboard sign
- the blue basket
- plastic basket
- the persimmon
- a white basket
- blue bowl
- cardboard box
- mango, bread bag, a small glass bowl, ...
- blue carpet
- black display stand
- a MacBook
- white iPhone
- a vibrant orange 1973 Plymouth Barracuda
- person's left hand

**Figure 6** Example video (top) and images (bottom) from SA-Co with annotated phrases and instance masks/IDs.

**Data Engine Components (Fig. 5).** Media inputs (image or video) are mined from a large pool with the help of a curated ontology. An AI model proposes noun phrases (NPs) describing visual concepts, followed by another model (e.g., SAM 3) that generates candidate instance masks for each proposed NP. The proposed masks are verified by a two-step process: first, in *Mask Verification (MV)* annotators accept or reject masks based on their quality and relevance to the NP. Second, in *Exhaustivity Verification (EV)* annotators check if all instances of the NP have been masked in the input. Any media-NP pairs that did not pass the exhaustivity check are sent to a manual correction stage, where humans add, remove or edit masks (using SAM 1 in a browser based tool), or use “group” masks for small, hard to separate objects. Annotators may reject ungroundable or ambiguous phrases.

**Phase 1: Human Verification.** We first randomly sample images and NP proposal with a simple captioner and parser. The initial mask proposal model is SAM 2 prompted with the output of an off-the-shelf open-vocabulary detector, and initial verifiers are human. In this phase, we collected 4.3M image-NP pairs as the initial SA-Co/HQ dataset. We train SAM 3 on this data and use it as the mask proposal model for the next phase.

**Phase 2: Human + AI Verification.** In this next phase, we use human accept/reject labels from the MV and EV tasks collected in Phase 1 to fine-tune Llama 3.2 (Dubey et al., 2024) to create AI verifiers that *automatically* perform the MV and EV tasks. These models receive image-phrase-mask triplets and output multiple-choice ratings of mask quality or exhaustivity. This new auto-verification process allows our human effort to be focused on the most challenging cases. We continue to re-train SAM 3 on newly collected data and update it 6 times. As SAM 3 and AI verifiers improve, a higher proportion of labels are auto-generated, further accelerating data collection. The introduction of AI verifiers for MV and EV roughly doubles the data engine’s throughput *vs.* human annotators. We refer to §A.4 for detailed analysis of how AI verifiers improve the data engine’s throughput. We further upgrade the NP proposal step to a Llama-based pipeline that also proposes hard negative NPs adversarial to SAM 3. Phase 2 adds 122M image-NP pairs to SA-Co/HQ.

**Phase 3: Scaling and Domain Expansion.** In the third phase, we use AI models to mine increasingly challenging cases and broaden domain coverage in SA-Co/HQ to 15 datasets (Fig. 15). A *domain* is a unique distribution of text and visual data. In new domains, the MV AI verifier performs well zero-shot, but the EV AI verifier needs to be improved with modest domain-specific human supervision. We also expand concept coverage to long-tail, fine-grained concepts by extracting NPs from the image alt-text where available and by mining concepts from a 22.4M node *SA-Co ontology* (§D.2) based on Wikidata (17 top-level categories, 72 sub-categories). We iterate SAM 3 training 7 times and AI verifiers 3 times, and add 19.5M image-NP pairs to SA-Co/HQ.**Phase 4: Video Annotation.** This phase extends the data engine to video. We use a mature image SAM 3 to collect targeted quality annotations that capture video-specific challenges. The data mining pipeline applies scene/motion filters, content balancing, ranking, and targeted searches. Video frames are sampled (randomly or by object density) and sent to the image annotation flow (from phase 3). *Masklets* (spatio-temporal masks) are produced with SAM 3 (now extended to video) and post-processed via deduplication and removal of trivial masks. Because video annotation is more difficult, we concentrate humans on likely failures by favoring clips with many crowded objects and tracking failures. The collected video data SA-Co/VIDEO consists of 52.5K videos and 467K masklets. See §D.6 for details.

## 5 Segment Anything with Concepts (SA-Co) Dataset

**Training Data.** We collect three *image datasets* for the PCS task: (i) SA-Co/HQ, the high-quality image data collected from the data engine in phases 1-4, (ii) SA-Co/SYN, a synthetic dataset of images labeled by a mature data engine (phase 3) without human involvement, and (iii) SA-Co/EXT, 15 external datasets that have instance mask annotations, enriched with hard negatives using our ontology pipeline. Notably in the SA-Co/HQ dataset we annotate 5.2M images and 4M unique NPs, making it the largest high-quality open-vocab segmentation dataset. We also annotate a *video dataset*, SA-Co/VIDEO, containing 52.5K videos and 24.8K unique NPs, forming 134K video-NP pairs. The videos on average have 84.1 frames at 6 fps. See §E.1 for details including full statistics, comparison with existing datasets and the distribution of concepts.

**SA-Co Benchmark.** The SA-Co evaluation benchmark has 207K unique phrases, 121K images and videos, and over 3M media-phrase pairs with hard negative labels to test open-vocabulary recognition. It has 4 splits: SA-Co/Gold has seven domains and each image-NP pair is annotated by three different annotators (used to measure human performance); SA-Co/Silver has ten domains and only one human annotation per image-NP pair; SA-Co/Bronze and SA-Co/Bio are nine existing datasets either with existing mask annotations or masks generated by using boxes as prompts to SAM 2. The SA-Co/VEval benchmark has three domains and one annotator per video-NP pair. See Tab. 28 for dataset statistics and Fig. 6 for example annotations.

**Metrics.** We aim to measure the usefulness of the model in downstream applications. Detection metrics such as average precision (AP) do not account for calibration, which means that models can be difficult to use in practice. To remedy this, we only evaluate predictions with confidence above 0.5, effectively introducing a threshold that mimics downstream usages and enforces good calibration. The PCS task can be naturally split into two sub-tasks, *localization* and *classification*. We evaluate localization using *positive micro F1* ( $pmF_1$ ) on positive media-phrase pairs with at least one ground-truth mask. Classification is measured with *image-level Matthews Correlation Coefficient* ( $IL\_MCC$ ) which ranges in  $[-1, 1]$  and evaluates binary prediction at the image level (“is the object present?”) without regard for mask quality. Our main metric, *classification-gated F1* ( $cgF_1$ ), combines these as follows:  $cgF_1 = 100 * pmF_1 * IL\_MCC$ . Full definitions are in §E.3.

**Handling Ambiguity.** We collect 3 annotations per NP on SA-Co/Gold. We measure *oracle* accuracy comparing each prediction to all ground truths and selecting the best score. See §E.3.

## 6 Experiments

We evaluate SAM 3 across image and video segmentation, few-shot adaptation to detection and counting benchmarks, and segmentation with complex language queries with SAM 3 + MLLM. We also show a subset of ablations, with more in §A. References, more results and details are in §F.

**Image PCS with Text.** We evaluate instance segmentation, box detection, and semantic segmentation on external and our benchmarks. SAM 3 is prompted with a single NP at a time, and predicts instance masks, bounding boxes, or semantic masks. As baselines, we evaluate OWLv2, GroundingDino (gDino), and LLMDet on box detection, and prompt SAM 1 with their boxes to evaluate segmentation. We also compare to APE, DINO-X, and Gemini 2.5 Flash, a generalist LLM. Tab. 1 shows that zero-shot, SAM 3 sets a new state-of-the-art on closed-vocabulary COCO, COCO-O and on LVIS boxes, and is significantly better on LVIS masks. On open-vocabulary SA-Co/Gold SAM 3 achieves *more than double* the  $cgF_1$  score of the strongest baseline OWLv2\*, and 74% of the estimated human performance. The improvements are even higher on the other SA-Co splits. Open vocabulary semantic segmentation results on ADE-847, PascalConcept-59, and Cityscapes show that SAM 3 outperforms APE, a strong specialist baseline. See §F.1 for details.<table border="1">
<thead>
<tr>
<th rowspan="3">Model</th>
<th colspan="6">Instance Segmentation</th>
<th colspan="8">Box Detection</th>
<th colspan="3">Semantic Segmentation</th>
</tr>
<tr>
<th colspan="2">LVIS</th>
<th colspan="4">SA-Co</th>
<th colspan="2">LVIS</th>
<th colspan="2">COCO</th>
<th colspan="4">SA-Co</th>
<th>ADE-847</th>
<th>PC-59</th>
<th>Cityscapes</th>
</tr>
<tr>
<th>cgF<sub>1</sub></th>
<th>AP</th>
<th>Gold</th>
<th>Silver</th>
<th>Bronze</th>
<th>Bio</th>
<th>cgF<sub>1</sub></th>
<th>AP</th>
<th>AP</th>
<th>AP<sub>o</sub></th>
<th>Gold</th>
<th>Silver</th>
<th>Bronze</th>
<th>Bio</th>
<th>mIoU</th>
<th>mIoU</th>
<th>mIoU</th>
</tr>
<tr>
<th></th>
<th></th>
<th></th>
<th>cgF<sub>1</sub></th>
<th>cgF<sub>1</sub></th>
<th>cgF<sub>1</sub></th>
<th>pmF<sub>1</sub></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>cgF<sub>1</sub></th>
<th>cgF<sub>1</sub></th>
<th>cgF<sub>1</sub></th>
<th>pmF<sub>1</sub></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>Human</td>
<td>—</td>
<td>—</td>
<td>72.8</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>74.0</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>OWLv2</td>
<td>20.1</td>
<td>—</td>
<td>17.3</td>
<td>7.6</td>
<td>3.9</td>
<td>0.64</td>
<td>19.9</td>
<td>35.2</td>
<td>38.2</td>
<td>42.4</td>
<td>16.9</td>
<td>7.1</td>
<td>4.1</td>
<td>0.95</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>OWLv2*</td>
<td>29.3</td>
<td>43.4</td>
<td>24.6</td>
<td>11.5</td>
<td>11.7</td>
<td>0.04</td>
<td>30.2</td>
<td>45.5</td>
<td>46.1</td>
<td>23.9</td>
<td>24.5</td>
<td>11.0</td>
<td>12.0</td>
<td>0.08</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>gDino-T</td>
<td>14.7</td>
<td>—</td>
<td>3.3</td>
<td>2.7</td>
<td>7.0</td>
<td>0.34</td>
<td>15.1</td>
<td>20.5</td>
<td>45.7</td>
<td>35.3</td>
<td>3.4</td>
<td>2.5</td>
<td>7.6</td>
<td>0.35</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>LLMDet-L</td>
<td>35.1</td>
<td>36.3</td>
<td>6.5</td>
<td>7.1</td>
<td>12.5</td>
<td>0.15</td>
<td>39.3</td>
<td>42.0</td>
<td>55.6</td>
<td>49.8</td>
<td>6.8</td>
<td>6.7</td>
<td>14.0</td>
<td>0.17</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>APE-D*</td>
<td>—</td>
<td>53.0<sup>†</sup></td>
<td>16.4</td>
<td>7.3</td>
<td>12.4</td>
<td>0.00</td>
<td>—</td>
<td>59.6<sup>†</sup></td>
<td>58.3<sup>†</sup></td>
<td>—</td>
<td>17.3</td>
<td>7.7</td>
<td>14.3</td>
<td>0.00</td>
<td>9.2<sup>†</sup></td>
<td>58.5<sup>†</sup></td>
<td>44.2<sup>†</sup></td>
</tr>
<tr>
<td>DINO-X</td>
<td>—</td>
<td>38.5<sup>†</sup></td>
<td>21.3<sup>δ</sup></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>52.4<sup>†</sup></td>
<td>56.0<sup>†</sup></td>
<td>—</td>
<td>22.5<sup>δ</sup></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>Gemini 2.5</td>
<td>13.4</td>
<td>—</td>
<td>13.0</td>
<td>8.3</td>
<td>7.3</td>
<td>10.7</td>
<td>16.1</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>14.4</td>
<td>9.4</td>
<td>8.2</td>
<td>12.4</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><b>SAM 3</b></td>
<td><b>37.2</b></td>
<td><b>48.5</b></td>
<td><b>54.1</b></td>
<td><b>49.6</b></td>
<td><b>42.6</b></td>
<td><b>55.4</b></td>
<td><b>40.6</b></td>
<td><b>53.6</b></td>
<td><b>56.4</b></td>
<td><b>55.7</b></td>
<td><b>55.7</b></td>
<td><b>50.0</b></td>
<td><b>47.1</b></td>
<td><b>56.3</b></td>
<td><b>13.8</b></td>
<td><b>60.8</b></td>
<td><b>65.2</b></td>
</tr>
</tbody>
</table>

**Table 1** Evaluation on image concept segmentation with text. AP<sub>o</sub> corresponds to COCO-O accuracy, \*: partially trained on LVIS, †: from original papers, δ: from DINO-X API. Gray numbers indicate usage of respective closed set training data (LVIS/COCO). See §F.1 for more baselines and results and §E.4 for details of human performance.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">ODinW13</th>
<th colspan="2">RF-100VL</th>
<th colspan="4">COCO</th>
<th colspan="4">LVIS</th>
<th colspan="4">ODinW13</th>
</tr>
<tr>
<th>AP<sub>0</sub></th>
<th>AP<sub>10</sub></th>
<th>AP<sub>0</sub></th>
<th>AP<sub>10</sub></th>
<th>AP</th>
<th>AP<sup>+</sup></th>
<th>AP<sup>+</sup></th>
<th>AP<sup>+</sup></th>
<th>AP</th>
<th>AP<sup>+</sup></th>
<th>AP<sup>+</sup></th>
<th>AP<sup>+</sup></th>
<th>AP</th>
<th>AP<sup>+</sup></th>
<th>AP<sup>+</sup></th>
<th>AP<sup>+</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td>Gemini2.5-Pro</td>
<td>33.7</td>
<td>—</td>
<td>11.6</td>
<td>9.8</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>gDino-T</td>
<td>49.7</td>
<td>—</td>
<td><b>15.7</b></td>
<td>33.7</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>gDino1.5-Pro</td>
<td>58.7</td>
<td>67.9</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><b>SAM 3</b></td>
<td><b>61.0</b></td>
<td><b>71.8</b></td>
<td>15.2</td>
<td><b>36.5</b></td>
<td><b>56.4</b></td>
<td><b>58.8</b></td>
<td><b>76.8</b></td>
<td><b>78.1</b></td>
<td><b>52.4</b></td>
<td><b>54.7</b></td>
<td><b>76.0</b></td>
<td><b>78.4</b></td>
<td><b>61.1</b></td>
<td><b>63.1</b></td>
<td><b>82.2</b></td>
<td><b>81.8</b></td>
</tr>
</tbody>
</table>

**Table 2** Zero-shot and 10-shot transfer on in-the-wild datasets.

**Table 3** Prompting with 1 exemplar on COCO, LVIS and ODinW13. Evaluation per prompt type: T (text-only), I (image-only), and T+I (combined text and image). AP<sup>+</sup> is evaluated only on positives examples.

**Few-Shot Adaptation.** We evaluate zero- and few-shot transfer of SAM 3 on ODinW13 and RF100-VL, with their original labels as prompts. We *do not* perform any prompt tuning. We fine-tune SAM 3 without mask loss, and report average bbox mAP in Tab. 2. SAM 3 achieves state-of-the-art 10-shot performance, surpassing in-context prompting in Gemini and object detection experts (gDino); more details in §F.3. RF-100VL contains domains with specialized prompts that are out of SAM 3’s current scope, but SAM 3 adapts through fine-tuning more efficiently than baselines.

**PCS with 1 Exemplar.** We first evaluate image exemplars using a single input box sampled at random from the ground truth. This can be done only on “*positive*” data, where each prompted object appears in the image. We report the corresponding AP<sup>+</sup> in Tab. 3 across three settings: text prompt (T), exemplar image (I), and both text and image (T+I); SAM 3 outperforms prior state-of-the-art T-Rex2 by a healthy margin on COCO (+18.3), LVIS (+10.3), and ODinW (+20.5). See §F.2 for more details and results on SA-Co/Gold.

**PCS with K Exemplars.** Next, we evaluate SAM 3 in an interactive setting, simulating collaboration with a human annotator. Starting with a text prompt, we iteratively add one exemplar prompt at a time: missed ground truths are candidate positive prompts, false positive detections are candidate negative prompts. Results (Fig. 7) are compared to a perfect PVS baseline, where we simulate the user manually fixing errors using ideal box-to-mask corrections. SAM 3’s PCS improves cgF<sub>1</sub> more quickly, as it generalizes from exemplars (e.g., detecting or suppressing similar objects), while PVS only corrects individual instances. After 3 clicks, interactive PCS outperforms text-only by +21.6 cgF<sub>1</sub> points and PVS refinement by +2.0. Performance plateaus after 4 clicks, as exemplars cannot fix poor-quality masks. Simulating a *hybrid* switch to PVS at this point yields gains, showing *complementary*.**Object Counting.** We evaluate on object counting benchmarks CountBench and PixMo-Count to compare with several MLLMs using Accuracy (%) and Mean Absolute Error (MAE) from previous technical reports and our own evaluations. See Tab. 4 for results and §F.4 for more evaluation details. Compared to MLLMs, SAM 3 not only achieves good object counting accuracy, but also provides object segmentation that most MLLMs cannot provide.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">CountBench</th>
<th colspan="2">PixMo-Count</th>
</tr>
<tr>
<th>MAE ↓</th>
<th>Acc ↑</th>
<th>MAE ↓</th>
<th>Acc ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>DINO-X</td>
<td>0.62</td>
<td>82.9</td>
<td><b>0.21</b></td>
<td>85.0</td>
</tr>
<tr>
<td>Qwen2-VL-72B</td>
<td>0.28</td>
<td>86.7</td>
<td>0.61</td>
<td>63.7</td>
</tr>
<tr>
<td>Molmo-72B</td>
<td>0.27</td>
<td>92.4</td>
<td>0.17</td>
<td>88.8</td>
</tr>
<tr>
<td>Gemini 2.5 Pro</td>
<td>0.24</td>
<td>92.4</td>
<td>0.38</td>
<td>78.2</td>
</tr>
<tr>
<td><b>SAM 3</b></td>
<td><b>0.12</b></td>
<td><b>93.8</b></td>
<td><b>0.21</b></td>
<td><b>86.2</b></td>
</tr>
</tbody>
</table>

**Table 4** Accuracy on counting benchmarks. Gray indicates usage of training sets.

**Video PCS with Text.** We evaluate video segmentation with text prompts on both our SA-Co/VEval benchmark and existing public benchmarks. For SA-Co/VEval, we report cgF<sub>1</sub> and pHOTA metrics (defined in §F.5) across its subsets (SA-V, YT-Temporal-1B, SmartGlasses). For public benchmarks, we use their official metrics. Baselines include GLEE, an open-vocabulary image and video segmentation model, “LLMDet + SAM 3 Tracker” (replacing our detector with LLMDet), and “SAM 3 Detector + T-by-D” (replacing our tracker with an association module based on the tracking-by-detection paradigm). In Tab. 5, SAM 3 largely outperforms these baselines, especially on benchmarks with a very large number of noun phrases. On SA-Co/VEval it reaches over 80% of human pHOTA. See §F.5 for more details.

<table border="1">
<thead>
<tr>
<th rowspan="3">Model</th>
<th colspan="6">SA-Co/VEval benchmark test split</th>
<th colspan="4">Public benchmarks</th>
</tr>
<tr>
<th colspan="2">SA-V<br/>(2.0K NPs)</th>
<th colspan="2">YT-Temporal-1B<br/>(1.7K NPs)</th>
<th colspan="2">SmartGlasses<br/>(2.4K NPs)</th>
<th>LVVIS<br/>(1.2K NPs)</th>
<th>BURST<br/>(482 NPs)</th>
<th>YTVIS21<br/>(40 NPs)</th>
<th>OVIS<br/>(25 NPs)</th>
</tr>
<tr>
<th>cgF<sub>1</sub></th>
<th>pHOTA</th>
<th>cgF<sub>1</sub></th>
<th>pHOTA</th>
<th>cgF<sub>1</sub></th>
<th>pHOTA</th>
<th>test mAP</th>
<th>test HOTA</th>
<th>val mAP</th>
<th>val mAP</th>
</tr>
</thead>
<tbody>
<tr>
<td>Human</td>
<td>53.1</td>
<td>70.5</td>
<td>71.2</td>
<td>78.4</td>
<td>58.5</td>
<td>72.3</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td>GLEE<sup>†</sup> (all NPs at once)</td>
<td>0.1</td>
<td>8.7</td>
<td>1.6</td>
<td>16.7</td>
<td>0.0</td>
<td>4.7</td>
<td>20.8</td>
<td>28.4</td>
<td><b>62.2</b></td>
<td>38.7</td>
</tr>
<tr>
<td>GLEE<sup>†</sup> (one NP at a time)</td>
<td>0.1</td>
<td>11.8</td>
<td>2.2</td>
<td>18.9</td>
<td>0.1</td>
<td>5.6</td>
<td>9.3</td>
<td>20.2</td>
<td>56.5</td>
<td>32.4</td>
</tr>
<tr>
<td>LLMDet<sup>†</sup> + <b>SAM 3</b> Tracker</td>
<td>2.3</td>
<td>30.1</td>
<td>8.0</td>
<td>37.9</td>
<td>0.3</td>
<td>18.6</td>
<td>15.2</td>
<td>33.3</td>
<td>31.3</td>
<td>20.4</td>
</tr>
<tr>
<td><b>SAM 3</b> Detector + T-by-D</td>
<td>25.7</td>
<td>55.7</td>
<td>47.6</td>
<td>68.2</td>
<td>29.7</td>
<td>60.0</td>
<td>35.9</td>
<td>39.7</td>
<td>56.5</td>
<td>55.1</td>
</tr>
<tr>
<td><b>SAM 3</b></td>
<td><b>30.3</b></td>
<td><b>58.0</b></td>
<td><b>50.8</b></td>
<td><b>69.9</b></td>
<td><b>36.4</b></td>
<td><b>63.6</b></td>
<td><b>36.3</b></td>
<td><b>44.5</b></td>
<td>57.4</td>
<td><b>60.5</b></td>
</tr>
</tbody>
</table>

**Table 5** Video PCS from a text prompt (open-vocabulary video instance segmentation) on SA-Co/VEval and public benchmarks (see Tab. 39 for more results and analyses). SAM 3 shows strong performance, especially on benchmarks with a large number of NPs. †: GLEE and LLMDet do not perform well zero-shot on SA-Co/VEval.

**PVS.** We evaluate SAM 3 on a range of visual prompting tasks, including Video Object Segmentation (VOS) and interactive image segmentation. Tab. 6 compares SAM 3 to recent state-of-the-art methods on the VOS task. SAM 3 achieves significant improvements over SAM 2 on most benchmarks, particularly on the challenging MOSEv2 dataset, where SAM 3 outperforms prior work by 6.5 points. For the interactive image segmentation task, we evaluate SAM 3 on the 37 datasets benchmark introduced in Ravi et al. (2024). As shown in Tab. 7, SAM 3 outperforms SAM 2 on average mIoU. See also §F.6 and Fig. 21 for interactive video segmentation.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4">J&amp;F</th>
<th>G</th>
<th>J&amp;F</th>
<th rowspan="2">Model</th>
<th colspan="4">Avg. mIoU</th>
</tr>
<tr>
<th>MOSEv1<br/>val</th>
<th>DAVIS17<br/>val</th>
<th>LVOSv2<br/>val</th>
<th>SA-V<br/>val</th>
<th>SA-V<br/>test</th>
<th>YTVOS19<br/>val</th>
<th>MOSEv2<br/>val</th>
<th>1-click</th>
<th>3-clicks</th>
<th>5-clicks</th>
<th>FPS</th>
</tr>
</thead>
<tbody>
<tr>
<td>SAMURAI</td>
<td>72.6</td>
<td>89.9</td>
<td>84.2</td>
<td>79.8</td>
<td>80.0</td>
<td>88.3</td>
<td>51.1</td>
<td>SAM 1 H</td>
<td>58.5</td>
<td>77.0</td>
<td>82.1</td>
<td>41.0</td>
</tr>
<tr>
<td>SAM2Long</td>
<td>75.2</td>
<td>91.4</td>
<td>85.9</td>
<td>81.1</td>
<td>81.2</td>
<td>88.7</td>
<td>51.5</td>
<td>SAM 2.1 L</td>
<td><b>66.4</b></td>
<td>80.3</td>
<td>84.3</td>
<td><b>93.0</b></td>
</tr>
<tr>
<td>SeC</td>
<td>75.3</td>
<td>91.3</td>
<td>86.5</td>
<td>82.7</td>
<td>81.7</td>
<td>88.6</td>
<td>53.8</td>
<td><b>SAM 3</b></td>
<td>66.1</td>
<td><b>81.3</b></td>
<td><b>85.1</b></td>
<td>43.5</td>
</tr>
<tr>
<td>SAM 2.1 L</td>
<td>77.9</td>
<td>90.7</td>
<td>79.6</td>
<td>77.9</td>
<td>78.4</td>
<td>89.3</td>
<td>47.9<sup>†</sup></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td><b>SAM 3</b></td>
<td><b>78.4</b></td>
<td><b>92.2</b></td>
<td><b>88.5</b></td>
<td><b>83.5</b></td>
<td><b>84.4</b></td>
<td><b>89.7</b></td>
<td><b>60.3</b></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

**Table 6** SAM 3 improves over SAM 2 in VOS. †: Zero-shot.

**Table 7** Interactive image segmentation on the SA-37 benchmark.

**SAM 3 Agent.** We experiment with an MLLM that uses SAM 3 as a tool to segment more complex text queries (see Fig. 25). The MLLM proposes noun phrase queries to prompt SAM 3 and analyzes the returned masks, iterating until the masks are satisfactory. Tab. 8 shows that this “SAM 3 Agent” evaluated zero-shot on ReasonSeg and OmniLabel surpasses prior work without training on any referring expression segmentation or reasoning segmentation data. SAM 3 Agent also outperforms previous zero-shot results on RefCOCO+ and RefCOCOg. SAM 3 can be combined with various MLLMs, with the same set of the system prompts for all those MLLMs, showing SAM 3’s robustness. See §G for more details.<table border="1">
<thead>
<tr>
<th rowspan="3">Model</th>
<th rowspan="3">MLLM</th>
<th colspan="4">ReasonSeg (gIoU)</th>
<th colspan="4">Omnilabel (AP)</th>
</tr>
<tr>
<th>val</th>
<th colspan="3">test</th>
<th colspan="4">val 2023</th>
</tr>
<tr>
<th>All</th>
<th>All</th>
<th>Short</th>
<th>Long</th>
<th>descr</th>
<th>descr-S</th>
<th>descr-M</th>
<th>descr-L</th>
</tr>
</thead>
<tbody>
<tr>
<td>X-SAM</td>
<td>Phi-3-3.8B</td>
<td>56.6</td>
<td>57.8</td>
<td>47.7</td>
<td>56.0</td>
<td>12.0*</td>
<td>17.1*</td>
<td>11.4*</td>
<td>8.8*</td>
</tr>
<tr>
<td>SegZero</td>
<td>Qwen2.5-VL 7B</td>
<td>62.6</td>
<td>57.5</td>
<td>—</td>
<td>—</td>
<td>13.5*</td>
<td>20.7*</td>
<td>12.4*</td>
<td>9.1*</td>
</tr>
<tr>
<td>RSVP</td>
<td>GPT-4o</td>
<td>64.7</td>
<td>55.4</td>
<td>61.9</td>
<td>60.3</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td colspan="2">Overall state-of-the-art<sup>†</sup></td>
<td>65.0</td>
<td>61.3</td>
<td>55.4</td>
<td>63.2</td>
<td>36.5</td>
<td>54.4</td>
<td>33.2</td>
<td>25.5</td>
</tr>
<tr>
<td><b>SAM 3</b> Agent</td>
<td>Qwen2.5-VL 7B</td>
<td>62.2</td>
<td>63.0</td>
<td>59.4</td>
<td>64.1</td>
<td>36.7</td>
<td>52.6</td>
<td>34.3</td>
<td>26.6</td>
</tr>
<tr>
<td><b>SAM 3</b> Agent</td>
<td>Llama4 Maverick</td>
<td>68.5</td>
<td>67.1</td>
<td>66.8</td>
<td>67.2</td>
<td>32.8</td>
<td>43.7</td>
<td>30.9</td>
<td>27.5</td>
</tr>
<tr>
<td><b>SAM 3</b> Agent</td>
<td>Qwen2.5-VL 72B</td>
<td>74.6</td>
<td>70.8</td>
<td>70.3</td>
<td>71.0</td>
<td>42.0</td>
<td><b>56.0</b></td>
<td>40.4</td>
<td>33.2</td>
</tr>
<tr>
<td><b>SAM 3</b> Agent</td>
<td>Gemini 2.5 Pro</td>
<td><b>77.0</b></td>
<td><b>74.0</b></td>
<td><b>75.8</b></td>
<td><b>73.4</b></td>
<td><b>45.3</b></td>
<td>53.8</td>
<td><b>45.1</b></td>
<td><b>37.7</b></td>
</tr>
</tbody>
</table>

**Table 8** SAM 3 Agent results. Gray indicates fine-tuned results on ReasonSeg (train), \* indicates reproduced results, underline indicates the main metric. †: LISA-13B-LLaVA1.5 for ReasonSeg; REAL for OmniLabel.

<table border="1">
<thead>
<tr>
<th></th>
<th>cgF<sub>1</sub></th>
<th>IL_MCC</th>
<th>pmF<sub>1</sub></th>
<th>#/img</th>
<th>cgF<sub>1</sub></th>
<th>IL_MCC</th>
<th>pmF<sub>1</sub></th>
<th>EXT</th>
<th>SYN</th>
<th>HQ</th>
<th>cgF<sub>1</sub></th>
<th>IL_MCC</th>
<th>pmF<sub>1</sub></th>
<th>Model</th>
<th>cgF<sub>1</sub></th>
<th>IL_MCC</th>
<th>pmF<sub>1</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>×</td>
<td>50.7</td>
<td>0.77</td>
<td><b>65.4</b></td>
<td>0</td>
<td>28.3</td>
<td>0.44</td>
<td>62.4</td>
<td>✓</td>
<td>×</td>
<td>×</td>
<td>23.7</td>
<td>0.46</td>
<td>50.4</td>
<td>Human</td>
<td>72.8</td>
<td>0.94</td>
<td>77.0</td>
</tr>
<tr>
<td>✓</td>
<td><b>52.2</b></td>
<td><b>0.82</b></td>
<td>63.4</td>
<td>5</td>
<td>39.4</td>
<td>0.62</td>
<td><b>62.9</b></td>
<td>✓</td>
<td>✓</td>
<td>×</td>
<td>32.8</td>
<td>0.57</td>
<td>56.9</td>
<td>SAM 3</td>
<td>54.0</td>
<td>0.82</td>
<td>65.9</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td>15</td>
<td>41.8</td>
<td>0.67</td>
<td>62.4</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>45.5</td>
<td>0.71</td>
<td><b>64.0</b></td>
<td>+ EV AI</td>
<td>61.2</td>
<td>0.86</td>
<td>70.8</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td>30</td>
<td><b>43.0</b></td>
<td><b>0.68</b></td>
<td>62.8</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td><b>47.4</b></td>
<td><b>0.74</b></td>
<td>63.8</td>
<td>+ MV AI</td>
<td><b>62.3</b></td>
<td><b>0.87</b></td>
<td><b>71.1</b></td>
</tr>
</tbody>
</table>

(a) Presence head.

(b) Hard Negatives.

(c) Training data.

(d) SAM 3 + AI verifiers.

**Table 9** Selected model and data ablations on SA-Co/Gold. Numbers *across* tables are not directly comparable.

**Selected Ablations.** In Tab. 9 we report a subset of the more extensive ablations from §A. Note that the ablated models are from different training runs than the model evaluated above. The presence head boosts cgF<sub>1</sub> by +1.5 (9a), improving image-level recognition measured by IL\_MCC by +0.05. Tab. 9b shows that adding hard negatives significantly improves the model performance, most notably the image-level IL\_MCC from 0.44 to 0.68. Tab. 9c shows that synthetic (SYN) training data improves over the external (EXT) by +8.8 cgF<sub>1</sub> and our high-quality (HQ) annotations add +14.6 cgF<sub>1</sub> on top of this baseline. We present detailed data scaling laws of both types of data in §A.2, showing their effectiveness on both in-domain and out-of-domain test sets. In Tab. 9d, we show how AI verifiers can improve pseudo-labels. Replacing the presence score from SAM 3 with that score from the exhaustivity verification (EV) AI verifier boosts cgF<sub>1</sub> by +7.2. Using the mask verification (MV) AI verifier to remove bad masks adds another 1.1 points. Overall, AI verifiers close half of the gap between SAM 3’s and human performance.

**Domain adaptation ablation.** With domain-specific synthetic data generated by SAM 3 + AI verifiers, we show that one can significantly improve performance on a new domain *without any human annotation*. We hold out one of the SA-Co domains, “Food&drink”, from training SAM 3 and AI verifiers. We then use three variants of training data for the *novel* “Food&drink” domain: high-quality AI+human annotations as in SA-Co/HQ (referred to as **SA-Co/HQ-Food**), synthetic annotations as in SA-Co/SYN, using AI but no humans (**SA-Co/SYN-Food**), and pseudo-labels generated before the AI verification step, i.e. skipping both AI verifiers and humans (**PL-Food**). Fig. 8 plots performance on the “Food&drink” test set of the SA-Co/Gold benchmark as each type of training data is scaled up. We mix the domain specific data and high-quality general domain data at a 1:1 ratio. PL-Food provides some improvement compared to the baseline SAM 3 (zero-shot), but is far below the other variants due to its lower quality. HQ-Food and SYN-Food show similar scaling behavior, with SYN-Food slightly lower but eventually catching up, without incurring any human annotation cost. This points to a scalable way to improve performance on new data distributions. More details are in §A.3.

![Figure 8: Domain adaptation via synthetic data. A line graph showing cgF1 (new domain) on the y-axis (35 to 55) versus New domain data on the x-axis (1.5K to 750K). Four data series are plotted: PL-Food (blue dashed line with circles), SA-Co/SYN-Food (orange dashed line with circles), SA-Co/HQ-Food (teal dashed line with circles), and Teacher (purple dashed line with circles). The Teacher and SA-Co/HQ-Food lines are nearly identical, starting at ~45 and rising to ~55. The SA-Co/SYN-Food line starts at ~43 and rises to ~55. The PL-Food line starts at ~40 and rises to ~46. A pink dotted line at ~37 represents the Baseline.](36f7936d2dce21f9b0c8784e3e74ebfd_11_img.webp)

**Figure 8** Domain adaptation via synthetic data. Synthetic (SYN) data generated by SAM 3 + AI verifiers (teacher system) achieves similar scaling behavior as *human-annotated* (HQ) data.## 7 Related Work

**Promptable and Interactive Visual Segmentation.** SAM (Kirillov et al., 2023) introduces “promptable” image segmentation with interactive refinement. While the original task definition included text prompts, they were not fully developed. SAM 2 (Ravi et al., 2024) extended the promptable visual segmentation task to video, allowing refinement points on any frame. SAM 3 inherits geometry-based segmentation while extending to include text and image exemplar prompts to segment all instances of a concept in images and videos.

**Open-Vocabulary Detection and Segmentation in Images** exhaustively labels every instance of an open-vocabulary object category with a coarse bounding box (detection) or a fine-grained pixel mask (segmentation). Recent open-vocabulary (OV) detection (Gu et al., 2021; Minderer et al., 2022) and segmentation (Ding et al., 2022; Liang et al., 2023) methods leverage large-scale vision-language encoders such as CLIP (Radford et al., 2021) to handle categories described by arbitrary text, even those never seen during training. While DETR (Carion et al., 2020) is limited to a closed set of categories seen during training, MDETR (Kamath et al., 2021) evolves the approach to condition on raw text queries. Image exemplars used as prompts to specify the desired object category (e.g., DINOv (Li et al., 2023a), T-Rex2 (Jiang et al., 2024)) present a practical alternative to text, but fall short in conveying the abstract concept of objects as effectively as text prompts. We introduce a new benchmark for OV segmentation with  $> 100\times$  more unique concepts than prior work.

**Visual Grounding** localizes a language expression referring to a region of the image with a box or mask. (Plummer et al., 2020) introduces phrase detection as both deciding whether the phrase is relevant to an image and localizing it. GLIP (Li et al., 2022b) and GroundingDino (Liu et al., 2023) formulate object detection as phrase grounding, unifying both tasks during training. MQ-GLIP (Xu et al., 2023) adds image exemplars to text as queries. Building on this trend toward models supporting multiple tasks and modalities, GLEE (Wu et al., 2024a) allows text phrases, referring expressions, and visual prompts for category and instance grounding in both images and videos. Unlike SAM 3, GLEE does not support exemplars or interactive refinement. LISA (Lai et al., 2024) allows segmentation that requires reasoning, while OMG-LLaVa (Zhang et al., 2024a) and GLaMM (Rasheed et al., 2024) generate natural language responses interleaved with corresponding segmentation masks, with GLaMM accepting both textual and optional image prompts as input. Some general-purpose MLLMs can output boxes and masks (Gemini2.5 (Comanici et al., 2025)) or points (Molmo (Deitke et al., 2025)). SAM 3 can be used as a “vision tool” in combination with an MLLM (§6).

**Multi-Object Tracking and Segmentation** methods identify object instances in video and track them, associating each with a unique ID. In tracking-by-detection methods, detection is performed independently on each frame to produce boxes and confidence scores, followed by association of boxes using motion-based and appearance-based matching as in SORT (Bewley et al., 2016; Wojke et al., 2017), Tracktor (Bergmann et al., 2019), ByteTrack (Zhang et al., 2022c), SAM2MOT (Jiang et al., 2025), or OC-SORT (Cao et al., 2023). An alternative is an end-to-end trainable architecture that jointly detects and associates objects, e.g., TrackFormer (Meinhardt et al., 2022), TransTrack (Sun et al., 2020), or MOTR (Zeng et al., 2022). TrackFormer uses a DETR-like encoder-decoder that initializes new tracks from static *object queries* and auto-regressively follows existing tracks with identity-preserving *track queries*. A challenge with joint models is the conflict between detection and tracking (Feichtenhofer et al., 2017; Yu et al., 2023a), where one needs to focus on semantics while the other on disentangling identities, even if their spatial locations overlap over time. SAM 3 is a strong image detector tightly integrated into a tracker to segment concepts in videos.

## 8 Conclusion

We present Segment Anything with *Concepts*, enabling open-vocabulary text and image exemplars as prompts in interactive segmentation. Our principal contributions are: (i) introducing the PCS task and SA-Co benchmark, (ii) an architecture that decouples recognition, localization and tracking and extends SAM 2 to solve concept segmentation while retaining visual segmentation capabilities, (iii) a high-quality, efficient data engine that leverages the complimentary strengths of human and AI annotators. SAM 3 achieves state-of-the-art results, doubling performance over prior systems for PCS on SA-Co in images and videos. That said, our model has several limitations. For example, it struggles to generalize to out-of-domain terms, which could be mitigated by automatic domain expansion but requires extra training. We discuss this and other limitations of our model in §B. We believe SAM 3 and the SA-Co benchmark will be important milestones and pave the way for future research and applications in computer vision.## 9 Acknowledgements

We would like to thank the following people for their contributions to the SAM 3 project: Alex He, Alexander Kirillov, Alyssa Newcomb, Ana Paula Kirschner Mofarrej, Andrea Madotto, Andrew Westbury, Ashley Gabriel, Azita Shokpour, Ben Samples, Bernie Huang, Carleigh Wood, Ching-Feng Yeh, Christian Puhrsch, Claudette Ward, Daniel Bolya, Daniel Li, Facundo Figueroa, Fazila Vhora, George Orlin, Hanzi Mao, Helen Klein, Hu Xu, Ida Cheng, Jake Kinney, Jiale Zhi, Jo Sampaio, Joel Schlosser, Justin Johnson, Kai Brown, Karen Bergan, Karla Martucci, Kenny Lehmann, Maddie Mintz, Mallika Malhotra, Matt Ward, Michelle Chan, Michelle Restrepo, Miranda Hartley, Muhammad Maaz, Nisha Deo, Peter Park, Phillip Thomas, Raghu Nayani, Rene Martinez Doehner, Robbie Adkins, Ross Girshik, Sasha Mitts, Shashank Jain, Spencer Whitehead, Ty Toledano, Valentin Gabeur, Vincent Cho, Vivian Lee, William Ngan, Xuehai He, Yael Yungster, Ziqi Pang, Ziyi Dou, Zoe Quake. We also thank the IDEA team for granting us DINO-X and T-Rex2 access to benchmark them on the SA-Co/Gold dataset.

## Appendix

<table><tr><td><b>A</b></td><td><b>Ablations</b></td><td><b>13</b></td></tr><tr><td>A.1</td><td>Model Ablations</td><td>13</td></tr><tr><td>A.2</td><td>Image Training Data Ablations</td><td>14</td></tr><tr><td>A.3</td><td>Automatic Domain Adaptation</td><td>16</td></tr><tr><td>A.4</td><td>Image Data Engine Annotation Speed</td><td>17</td></tr><tr><td>A.5</td><td>Video Data Engine Annotation Speed</td><td>18</td></tr><tr><td>A.6</td><td>Video Training Data Ablations</td><td>18</td></tr><tr><td><b>B</b></td><td><b>Limitations</b></td><td><b>18</b></td></tr><tr><td><b>C</b></td><td><b>Model Details</b></td><td><b>19</b></td></tr><tr><td>C.1</td><td>Model Architecture</td><td>19</td></tr><tr><td>C.2</td><td>Image Implementation Details</td><td>19</td></tr><tr><td>C.3</td><td>Video Implementation Details</td><td>22</td></tr><tr><td>C.4</td><td>Model Training</td><td>23</td></tr><tr><td><b>D</b></td><td><b>Data Engine Details</b></td><td><b>26</b></td></tr><tr><td>D.1</td><td>Media Pool</td><td>26</td></tr><tr><td>D.2</td><td>SA-Co Ontology</td><td>26</td></tr><tr><td>D.3</td><td>Phase 1: Human Verification</td><td>26</td></tr><tr><td>D.4</td><td>Phase 2: Human + AI Verification</td><td>28</td></tr><tr><td>D.5</td><td>Phase 3: Scaling and Domain Expansion</td><td>32</td></tr><tr><td>D.6</td><td>Phase 4: Video Annotation</td><td>32</td></tr><tr><td><b>E</b></td><td><b>SA-Co Dataset and Metric Details</b></td><td><b>34</b></td></tr><tr><td>E.1</td><td>SA-Co Training Data</td><td>34</td></tr><tr><td>E.2</td><td>SA-Co Evaluation Benchmark</td><td>35</td></tr><tr><td>E.3</td><td>Metrics</td><td>37</td></tr><tr><td>E.4</td><td>Human Performance on SA-Co</td><td>37</td></tr><tr><td>E.5</td><td>Additional Dataset Examples</td><td>39</td></tr></table>