# Research Questions and Analysis

## 1. Why did you pick the particular project?

This project in itself provided me an opportunity to work with deep reinforcement learning, something which I have been fascinated about but have not actually had a good chance to work on, a chance to explore variational auto-encoders, and most interesting of all was the representational learning aspect with which I was able to experiment a lot, and I will continue to do so. It was a great to learn how we individually build up and combine so many parts such as the DQN Agents, the VAEs, the World Models/Value Models, their nuances, thinking about which works then testing and iterating over results to see how we can do better has certainly helped me judge my research acumen as well as my appetite for research. Most of all, as my projects might have already suggested enabling edge device or CPU inference or even reducing the model sizes, using quantization and their research is something I am very very fascinated by and thus experimenting on the VQVAE and also trying it on a whole different game with a lot of different nuances to it provided me a very good opportunity to experiment to my heart's content.

---

## 2. If you had more compute/time, what would you have done?

If I had more time, I would've done a lot of things with this project, it is best to write in a bulleted form taking it from the top : 
- As I suggested in the DQN Agent training part, due to the constraints(compute/time) I didn't really get to explore a bunch of different methods for training even better agents than I was able to during the experiment, as the agents are used to generate the game data, I realized that I had to be efficient and quick to move towards VQVAE training which understandably took a bunch of time and compute already by itself. I made naïve decisions in the parts where I generated the data and simply went with it, without being able to generate whole sets of data and evaluating which data was the best for the VQVAE training, so if I could I would like to experiment with the data generation and the model building further, I was not able to train a DuelingDQN, a DuelingDQN with LSTMs, GRUs or maybe some way to add attention so that the agent has an understanding of the temporal aspects of a game such as Ms.PacMan.

- Further, I was not able to train a very dense model which could have been able to add to the overall better frame reconstructions (if you would look at the frames you could see that they are a bit blurry), however I believe the way I was able to reduce the latent and then work with the reduced embedding dimension and codebook and then being able to train a model on it was a good experience, I have a direction now and understand what to and not to do.

- Also, truthfully I could not give a lot of thought to the the World Model, rather than using the decoder as said in the docs, I tried to go and train a GRU based model to make up for the temporal aspects, I would like to overall do better at the Value Model as well.

- And finally, I would run lots and lots of ablations on the same, use saliency maps on a set of trained VQ-VAEs under different scenarios to see which VQ-VAE prioritizes which activations, I still need to understand why the VQVAE specializes in some areas and neglects the others, I would run ablation on PER, Non-PER or even Uniform Replay Buffer for a longer time and see how each affects it etc. I have lots and lots of questions which I still need answers to.

---

## 3. What did you learn in the project?

I learned a lot throughout the project, as I mentioned before I had never worked with Deep RL or even with VAE, I had experiences with Auto Encoders but not specially with VAEs or VQ-VAEs, this project really helped me to explore a lot throughout the failures I encountered while trying to make all of the things to work, the experiments which I did with the DQNs, the VQVAEs, while they were really frustrating at times, helped me learn about what to do and what not to do in certain scenarios, I still believe that there is room for exploration in the latent state activations to understand a lot about my attempt at this project. I was able to explore how the VQ-VAEs behave in cases of information bottlenecks in complex dynamic environments, it taught me a lot about how the model starts to underutilize the latent space as indicated by the training run around the 125 epoch mark of the model on the 450K samples.

---

## 4. What surprised you the most?

I was surprised at how many nuances are present in the model training and optimization as well as how the model was able to work better under reduced latent space but didn't work better with FiLM! I implemented FiLM with a lot of hope but it completely fell apart at that moment!(There goes my 4 hours), but it was really interesting to see how the model was able to work significantly better in a reduced latent space rather than when given a much more bigger space. I was also really surprised when I read thorugh Paras' atari-pixels repository on github, it really intrigued me that we can essentially completely run a game through neural networks single handedly! This was also one of my more prominent motivations to go ahead with the project, but sadly I couldn't really do so well after all, but it was really fun and I am sure that if I had not taken the bold step of using a complex game such as Ms.PacMan with my miniscule compute budget and the time constraint I would definitely have done a better job.

---

## 5. If you had to write a paper on the project, what else needs to be done?

I think this project leaves a lot to be desired, what is happening in the reduced latent space that it is achieving a better result when being reduced, even when the odds are that some other techniques perform better. It gives hope for me to explore a lot of things. 

Some points which I think are necessary to explore before we proceed with writing a paper on this : 

- Analyze the theoretical foundations of why latent compression forces the model to focus on salient features and what information it chooses to discard.
- Run controlled ablations across multiple bottleneck sizes (8, 16, 32, 64, 128, 256, 512) to systematically map the compression-fidelity trade-off curve and identify the optimal compression point
- Develop quantifiable measures using gradient-based saliency maps and attention visualization to understand exactly which visual components the model prioritizes at different compression levels
- Introduce systematic perturbations (noise, occlusion, color changes) to test which components of the input most critically affect both reconstruction quality and downstream RL performance
- Cross-validate findings across multiple Atari games to ensure the compression benefits aren't specific to Ms. Pac-Man
- Investigate how the VQ-VAE approach scales to more complex scenarios, such as games with multiple levels or procedurally generated content where the model encounters novel visual patterns during deployment
- Explore whether attention mechanisms can further improve the model's ability to focus on task-relevant features while maintaining efficient compression

I have also maintained a log about my experiments in the [Part1](./part1_mspacman.md), [Part2](./part2_mspacman.md), [Part3](./part3_mspacman.md), [Part4](./part4_mspacman.md) and [Part5](./part5_mspacman.md). However, I might have written a lot more that I was expected to do about the experiments just to be clear and still have tried to be concise, I sincerely apologize if at times the report seems to feel like a chore to read through.

Thanks a lot for reading through!