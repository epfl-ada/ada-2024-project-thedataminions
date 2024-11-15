# “Are You Stuck in a YouTube Bubble? It's Time to Expand Your Information Sources!”
## Almost there !
Make sure to have all the tools before starting by running the following lines:
```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```
## Repository structure
- 📁 `data`: 
   - 📁 `psycache`: ?
   - `results.ipynb`: Main notebook for Milestone 2.
- :file_folder: `src`:
   - `utils.py`: Utilities functions used in the notebooks.
- `.gitignore`: Specifies files and folders to ignore.
- `README.md`: Main documentation file of the repository, providing an overview and general instructions.
- `pip_requirements.txt`: List of packages required for our code.
- `results.ipynb`: 

## Abstract
The common advice suggests that to stay well-informed, one should diversify its sources.  However, YouTube users often fall into clusters, groups formed by particularly active people within a channel. These clusters reflect communities invested into the same channel but don't yet imply that they don’t diversify their sources.
The problem arises when these clusters evolve into bubbles, where users become confined to content that aligns with their existing beliefs, limiting exposure to other points of view. In these bubbles, interaction is mostly with individuals who share the same belief, which can contribute to polarization and radicalization.
In this study, we aim to analyze how users in News and Politics content tend to restrict their interactions to specific channels (forming clusters) and whether these clusters lead to the creation of bubbles. We will first identify clusters based on commenting behavior and then assess how isolated these clusters are, focusing on whether they cross over into informational bubbles that reinforce one-sided perspectives.

## Research questions

## Additional data
To identify the leading news channels on Youtube, we are using following [article]( https://pressgazette.co.uk/media-audience-and-business-data/media_metrics/youtube-news-publishers-2023-gb-news-piers-morgan-cnn-fox/). We will focus on the TOP 5, excluding Vox (ranked 5th), as it was founded in 2014, which is not well representated in the timeframe of our data (2005-2019).

## Methods

### Part 1: Data handling and filtering
### Part 2: Defining clusters
### Part 3: Identifying bubbles
#### Compare the clusters of main channels to a random pool of users
#### Examine whether there are bubbles within a cluster
#### 
### Part 4: Analyzing bubbles
#### 
####


## Proposed timeline 
|Timeframe | Tasks | 
|--------|--------------|
|Week 10 | <ol><li>Pairwise overlap algorithm</li><li>Bubbles analysis resp. random users</li><li>Degree of isolation</li></ol>| 
|Week 11  | <ol><li>Bubbles identification</li><li>Plots for data visualization</li><li>Website familiarization and initiation</li></ol>| 
|Week 12      | <ol><li>Bubbles analysis and closeness</li><ol>| 
|Week 13      |<ol><li>Website development</li><li>Data story refining</li></ol>| 
|Week 14  | <ol><li>Website refining</li><li>Code and readme cleaning</li></ol>| 

## Internal milestone & team organization
- **22.11.2024**: week 10 tasks    (Mila, Lou-Anne, Andreas)
- **29.11.2024**: week 11 tasks   (Manon, Hortense, Mila)
- **06.12.2024**: week 12 tasks   (Lou-Anne, Andreas)
- **13.12.2024**: week 13 tasks   (Manon, Hortense)
- **20.12.2024**: week 14 tasks   (All)
