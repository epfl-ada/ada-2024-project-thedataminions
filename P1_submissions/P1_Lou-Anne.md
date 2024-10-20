IDEA N°1:


→The title of this project will be: "Viewer Errors in Channel Names - Discovery of New
Channels". Each channel has a name (stored as 'name_cc' in df_channels_en.tsv.gz). Some
channels share common parts in their names, such as "Thunder Play", "Thunder Prod", and
"Thunder Rain". An intriguing project idea is to detect increases for one channel ( in
df_timeseries_en.csv.gz) over a random period (e.g., 1 year), and investigate if other
channels with similar names also gain views. This analysis can also be extended to
subscribers - do the numbers of subscribers increase? The main questions are : do viewers
tend to confuse channel names, leading to unwanted videos views? Would it be smart for
new channels to name after famous ones? Will it increase views and visibility? The primary
aim is to identify viewer errors: instances where viewers intend to watch videos from a
specific growing channel, but inadvertently watch videos from another channel with a similar
name.


IDEA N°2:


→ The title will be : “American presidential election from 2005 to 2019 : does the result of the
election follow youtube data?”. The main goal of this project is to analyze the past
presidential elections winners and see if there is a relation to the numbers of
views/subscribers of related channels. In other words, it will be to identify channels related to
the American elections, focusing on channels of politics category (in
df_channels_en.tsv.gz), in channel names (name_cc) looking for keywords like ‘election’,
‘presidentiel’, ‘Obama’ for e.g. and retrieving channels name from specific videos with video
tags with these keywords (from dataset yt_metadata_en.jsonl.gz). We will find the date of
the American election from 2005 to 2019, take data from 2-3 months before, and try to look
at the increase in viewers / subscription for these channels. For each election, specific
keywords will be looked at depending on the candidates, and if we have an increase in
viewers/ subscribers for specific candidates/ or political orientation, we could try to see if
there is a correlation between the two (most viewers for videos related to ‘Vote for Trump’ =
more votes for him?). Our data analysis could be compared to the real result that happened,
to show if it’s working or not. It could even be a tool to predict the next election !
This could also be done for other elections, not only for Americans elections, and we could
compare the different craze of people around elections, depending on their nationality.


IDEA N°3:


The title will be ‘A secret to fame’. The goal of this data analysis will be to explore the
different patterns Youtube superstar took. Classing the Youtube star by categories of videos
(categories_cc in df_channels_en.tsv.gz) and then take the top 10 superstar of each
categories, and looking at the pattern thy took : how many videos in a week (delta_videos in
df_timeseries_en.tsv.gz)? How long (duration in yt_metadata_en.jsonl.gz)? When did they
post the video (upload_date in yt_metadata_en.jsonl.gz)? The final goal will be for the user
to ask our interactive site what category he/she wants to make videos about, and receive a
to-do list on the best way to post, how much time, what keywords to put in the description,
and even how much money he/she can expect to make in a given amount of time (taken into
account that every youtubers that make a certain number of views
