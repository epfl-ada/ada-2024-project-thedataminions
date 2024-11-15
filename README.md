# “Are You Stuck in a YouTube Bubble? It's Time to Expand Your Information Sources!”

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
- 📁 `data`: clean and filtered data saved in the repository
   - `dataset` 
- :file_folder: `src`:
   - `utils.py`: Helper functions used in the notebook.
- `.gitignore`: Specifies files and folders to ignore.
- `README.md`: Main documentation file of the repository, providing an overview and general instructions.
- `pip_requirements.txt`: List of packages required for our code.
- `results.ipynb`: Main work for Milestone 2.

## Abstract
The common advice suggests that to stay well-informed, one should diversify its sources. We can define clusters of youtube users, which formed by particularly active people within a channel. These clusters reflect communities invested into the same channel but don't yet imply that they don’t diversify their sources to retrieve information on a particular topic.

Problems can arise when these clusters evolve into bubbles, where users become confined to specific content, limiting exposure to other points of view. In these bubbles, interactions are mostly between individuals who share the same interests, whether it's the same video subjects or channels followed.

In this study, we aim to analyze whether users that frequently interact under News and politics content of a certain channel form a bubble. After identification of clusters based on commenting behavior in the main media channels, we will assess how isolated they are, focusing on whether they cross over into bubbles.


## Research questions
- Are clusters in our dataset indicative of bubbles with limited perspective diversity ?
- Do commenters of major news channels form bubbles, or do they behave similarly to random Youtube users ?
- How isolated are the bubbles within the News and Politics community ? Can we find reasons for this isolation ?
- For what purpose do the identified bubbles use News and Politics content (e.g politics, daily news, etc.) ? Are the bubbles isolated due to the media they refer to, or the common topics of interest ?

## Additional data
To identify the leading news channels on Youtube, we are using the following [article]( https://pressgazette.co.uk/media-audience-and-business-data/media_metrics/youtube-news-publishers-2023-gb-news-piers-morgan-cnn-fox/). We will focus on the TOP 5, excluding Vox (ranked 5th), as it was founded in 2014, which is not well representated in the timeframe of our data (2005-2019).

## Methods
In order to answer our questions, we will perform the following analyses. As you can see in results.ipynb, some of them have already been achieved (Part 1 and half of Part 2). 

### Part 1: Data handling and filtering
First, we will need to clean the dataset by removing any irrelevant data, such as NaNs and empty data. Next, we will filter the data to retain only those related to the “News & Politics” category across the several dataframes that we use : df_channels_en.tsv, yt_metadata_en.json, youtube_comments.tsv. Afterwards, we will isolate users who commented on the videos of the 5 largest news channels (see ‘Additional data’), later defined as ‘main channels’ : CNN, ABC News, BBC News, Al Jazeera English, Fox News.

### Part 2: Defining clusters
Next, we will define and isolate the clusters for analysis. To do so we will determine a threshold for the number of comments per user in order to retain only the most active commenters in each main channel. We will first plot the numbers of comments per user in each main channel to spot outliers. Then a histogram for each main channel will allow us to visualize the distribution of comments per user. Based on this, we will define a threshold for each main channel. Possible methods can be used such as choosing the median, the first or third percentile of the distribution as threshold or define it by visualization on the first set of plots done before. From that, we will create the clusters for each main channel by keeping users whose total number of comments is above the threshold.

### Part 3: Identifying bubbles
To identify bubbles within the News and Politics community, we will adopt two strategies :

<div align="center">
   <img src="src/figures/bubbles.jpg" alt="Bubbles identification diagram" width="60%">
</div>

- **Compare the clusters of main channels to random pool of Youtube users** (figure above, top part)\
We will see if the clusters identified for each main channel represent a bubble, i.e  that they are closer  in the content they comment on than a random sample of Youtube users representing a control sample. This will be achieved by comparing the pairwise overlap values of the videos on which the cluster users have commented, to the one of the control sample. This will help us see if particularly active people in a main channel are more likely to watch/comment on similar content than a random group of Youtube users. If we observe a different behavior between our clusters and the control sample, we can classify our cluster as a bubble.

- **Examine whether there are bubbles within a cluster** (figure above, bottom part)\
We will look for the existence of multiple bubbles into the same cluster, which could show us divergence between commenters of the same channel. To do so we will use a flat clustering method such as DBSCAN (from [week 9 course](https://docs.google.com/presentation/d/1OWnmnm8oHRBz1JWymxFBNlClpVLLY9ke/edit#slide=id.p45
)). Pairwise overlap between the users of the new bubbles will then measure their isolation from one another.

Network graphs will help us visualize the links between users, showing both their levels of connectivity and isolation. Furthermore, A Venn diagram will complement this by illustrating the size of each bubble as well as their isolation through the extent of users overlap. 

### Part 4: Bubble analysis
For each identified bubble, we will look at the following properties :

- **Define a metric for the bubbles’ isolation**\
We will assess the degree of isolation, which relies on the overlap in the commented videos between users of both the same and different bubbles. For example, if users within a bubble commented on the same videos, respectively on totally different videos than users of another bubble, the former can be considered as highly isolated. To do so, we will use pairwise overlap calculations again.

- **Are bubbles formed by the fact that users watch videos from the same channels, or of a same topic within News & Politics ?**\
To assess this, we will analyze the descriptions of the videos viewed by users within a bubble. We will count how many times each word is employed. Looking at their frequency and relevance we will see if a particular topic can be deduced. Searched topics will remain vast, i.e. political information, daily news, international news, case studies, etc. If common topics are found, we can conclude that the bubble is not only shaped by the users’ sources, but also by shared topic in News & Politics.

- **How closely is a bubble associated with a particular video or channel ?**\
For each video where a member of the bubble has commented, we will calculate the fraction of comments made by members of the same bubble. This is a measure for the “closeness of the video to the bubble”, which can then be expanded to encompass all videos on a channel, providing the “closeness of the channel to the bubble.”

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
