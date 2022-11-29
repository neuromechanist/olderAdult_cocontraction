# olderAdults_coContraction
Older adults increasingly suffer from challenges in balance during locomotion and complications from falling. Mechanical perturbations, as a proxy to the locomotion challenges, may help identify the response differences across aging and act as an exercise to improve the elderly’s fall risk.

We used perturbations in a seated exercise, recumbent stepping, to remove the burden of maintaining balance and to focus on the body’s response to the mechanical perturbations at different timing.

## Small background on recumbent stepping
Recumbent stepping is the arms and legs seated rhythmic task used to apply the perturbations. Recumbent stepping has only one degree of freedom, and similarly to walking, contralateral limbs are coupled together, e.g., the left arm and right leg would extend together. The advantages of this particular task include not requiring maintaining balance while sharing motor principles with gait and the ability to apply perturbations at different times and intensities with the possibility of translating to accessible devices, such as small stepping or cycling devices at homes or gyms.

## The perturbations
Perturbations used for this task was a brief 200 ms increase in stepping resistance over each stride. The resistance increase was in a way that would require the subjects to use three times the torque to drive the steeping device at the same speed. Subjects were asked to try to complete the exercise 1) smoothly and 2) follow the visual cue provided on the screen. We did not provide any feedback about how the subjects responded to the perturbations and followed the instructions. 

Subjects performed the stepping exercise for 10 minutes for each *condition* (more on that later). The beginning and final two minutes were perturbation free and called the *pre* and *post* blocks. The middle six minutes involved applying perturbations during each stride, except for a random 1 in 5 *catch* stride that did not involve perturbations. Each condition only involved one perturbation timing.

<img src="https://user-images.githubusercontent.com/44906843/204468955-ff4ae26f-8ec7-4a1d-8eb9-62811d5022a6.png" width="600">
<!-- ![perturbed-stepping task blocks](https://user-images.githubusercontent.com/44906843/204468955-ff4ae26f-8ec7-4a1d-8eb9-62811d5022a6.png =x300) -->


## The conditions
There were, overall, four conditions based on the time the perturbation was applied. We applied the perturbation during the lower-limb extension onset or mid-extension, totaling four conditions (left/right * mod-extension/extension-onset = four conditions). The instructions and the task goals were repeated before the start of each condition. The condition order was pseudo-randomized for every subject.

## Motor error analysis
We identified two types of motor errors based on the instructions provided to the subjects. The temporal error was the difference between the pace provided by the visual cue (1 ~stride~ per two seconds = 1 ~step~ per second) and the actual stepping duration for each stride. The spatial error was the difference between the time-normalized stepping profile between each stride and the averaged baseline stride during the *pre* block.

<img src="https://user-images.githubusercontent.com/44906843/204468829-09d4736e-e6b8-45f4-b835-1ac26723d19b.png" width="600">
<!-- ![motor errors during perturbed stepping](https://user-images.githubusercontent.com/44906843/204468829-09d4736e-e6b8-45f4-b835-1ac26723d19b.png =x300) -->

## Co-contraction analysis
Co-contraction is the concurrent activation of the functionally opposing muscles around a single joint. Co-contraction increases joint stiffness, which, in turn, increases the local stability of the joint. For example, if the arm’s biceps and triceps activate at the same time, the stiffness of the elbow would increase, and it can withstand stronger forces in random directions.

Co-contraction index (CI) quantifies co-contraction and is defined as twice the ratio of the antagonist muscle activation to the sum of agonist and antagonist muscle activations. Muscle activation is quantified based on each step’s functional agonist and antagonists. The agonist muscle for each step is the muscle that can drive stepping in the desired direction. The antagonist’s muscle is the muscle that would hinder or slow down the stepping in each step. Muscle activation is quantified as the area under the curve of the EMG envelope for each step. Therefore, each step would have one CI for each muscle pair. If CI < 1, the muscle pair would drive stepping, if CI > 1, the muscle pair would resist stepping, and if CI ~ 1, the muscle pair’s driving and resisting would likely be in balance, or both muscles were not significantly active.

## Code structure
The main pipeline is structured within four Jupyter notebooks. Steps 1 to 3 involve preprocessing the EMG data and quantifying CI and other metrics derived from CI. Only the code and the final figures are shared at this time.

Step 4 involves quantifying the motor errors. Similarly, only the code and the final fugues are shared. We hope to share the CI data once the paper is published.

Three Python modules are called within the notebooks. `EssentialEMGFuncs.py` includes preprocessing and processing functions to import, filter, and process EMG data to reach the CIs. `SMART_Funcs` is from the [SMART Toolbox](https://github.com/jonathanvanleeuwen/SMART) with minimal modifications. `lineNdots.py` is the plotting function based on the Seaborn scatter plots.
