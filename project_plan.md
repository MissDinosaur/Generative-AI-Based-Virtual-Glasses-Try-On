### Project Timeline (April–July 2025)
Project deadline: July 21, 2025

Start date: April 28, 2025

Total duration: about 12 weeks

#### Weekly plan
#### Phase 1: Foundation & Prototyping (Week 1–2)
**Week 1**
- Do the research
- Confirm the delivery requirement
- Make the project plan

**Week 2-3: Data Preparation**
- Confirm the technology stack
- Organize selfies & glasses datasets
- Label/annotate if needed
- Explore public datasets (e.g., CelebA)
- Implement face detection + landmark extraction

**Week 4: Face Alignment Module**

- Normalize facial orientation using landmarks
- Estimate head pose
- Save processed selfies

**Week 4-5: Design Glasses Overlay Pipeline**

- Align glasses to landmarks (2D)
- Visual check on overlay quality
- Start building image pair dataset (face + aligned glasses)

**Week 6-7: Model Selection & Setup**

- Choose generative model (e.g., ControlNet or Pix2PixHD)
- Set up training pipeline
- Train on your dataset or use transfer learning

**Week 8: Model Evaluation**

- Visual check of outputs
- Fine-tune model or try different inputs (e.g., sketch maps, edge maps)

**Week 9-10: Polishing**

- Postprocessing (filters, shadows, realism)
- Test on unseen data (more selfies)
- Optional: simple Gradio demo or Flask + HTML/JS (possibly choose Flask UI)

**Week 11: Submission and deployment**

- Submit/deliver project
- Prepare demo

**Week 12**
Prepare for Case-study final presentation
