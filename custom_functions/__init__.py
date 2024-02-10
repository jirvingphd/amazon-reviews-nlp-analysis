
def mute_color(color_name, saturation_adjustment=0.5, lightness_adjustment=1.2):
    """
    Mutes a given CSS color name.
    Source: Conversation with ChatGPT on 02/09/24

    Parameters:
    - color_name: The name of the CSS color to be muted.
    - saturation_adjustment: Factor to adjust the color's saturation. 
                              Values < 1 decrease saturation; values > 1 increase it.
    - lightness_adjustment: Factor to adjust the color's lightness.
                            Values < 1 make the color darker; values > 1 make it lighter.

    Returns:
    - muted_hex: The hexadecimal representation of the muted color.
    """
    import matplotlib.colors as mcolors

    # Convert CSS color name to RGB
    rgb = mcolors.to_rgb(color_name)
    
    # Convert RGB to HSL
    h, l, s = mcolors.rgb_to_hsv(rgb)
    
    # Adjust saturation and lightness
    adjusted_hls = (h, max(0, min(l * lightness_adjustment, 1)), max(0, min(s * saturation_adjustment, 1)))
    
    # Convert back to RGB
    adjusted_rgb = mcolors.hsv_to_rgb(adjusted_hls)
    
    # Convert RGB back to Hex for CSS
    adjusted_hex = mcolors.to_hex(adjusted_rgb)
    
    return adjusted_hex


def mute_colors_by_key(colors_dict, 
                       saturation_adj=.5,
                       lightness_adj=1.2, 
                       keys_to_mute = [],
                      colors_to_mute=[]):
    """
    Params:
    - colors_dict: Dictionary of css color names.
    - saturation_adj: Factor to adjust the color's saturation. 
                              Values < 1 decrease saturation; values > 1 increase it.
    - lightness_adj: Factor to adjust the color's lightness.
                            Values < 1 make the color darker; values > 1 make it lighter.
    - keys_to_mute: Either list of keys from colors_dict or None to apply to all colors
    - colors_to_mute: Color values in the dictionary to be muted.
    
    Exception raised if both keys_to_mute and colors_to_mute are empty.
    """
    if keys_to_mute is None:
        keys_to_mute = colors_dict.keys()
        
    if (len(keys_to_mute) == 0 ) & (len(colors_to_mute)==0):
        raise Exception("Must provie either keys_to_mute or colors_to_mute")
    # colors_to_mute = []
    muted_colors = {}
    for k,v in colors_dict.items():
        
        if (k in keys_to_mute) | (v in colors_to_mute):
            v = mute_color(v,
                             saturation_adjustment=saturation_adj,
                             lightness_adjustment=lightness_adj)
        muted_colors[k] = v
    return muted_colors



def get_ngram_measures_df(tokens, ngrams=2, measure='raw_freq', top_n=None, min_freq = 1,
                             words_colname='Words', return_finder_measures = False,
                         group_name = None, join_words = True, sep= " ", multi_index=False):
    """
    Return the desired ngrams dataframe of requested measure. 
    Alternatively, return the finder and measure classes.

    This function will be used as a helper function for a comparison function.
    """
    import pandas as pd

    import nltk
    if ngrams == 4:
        MeasuresClass = nltk.collocations.QuadgramAssocMeasures
        FinderClass = nltk.collocations.QuadgramCollocationFinder
        
    elif ngrams == 3: 
        MeasuresClass = nltk.collocations.TrigramAssocMeasures
        FinderClass = nltk.collocations.TrigramCollocationFinder
    else:
        MeasuresClass = nltk.collocations.BigramAssocMeasures
        FinderClass = nltk.collocations.BigramCollocationFinder

    # instantiate the selected MeasuresClass
    measures = MeasuresClass()
    # instantiate a Finder
    finder = FinderClass.from_words(tokens)
    # Apply frequency filtering
    finder.apply_freq_filter(min_freq)

    # Return finder and measures instead of 
    if return_finder_measures == True:
        return finder, measures

    # Select desired measure to return
    selected_measure = getattr(measures, measure)
    scored_ngrams = finder.score_ngrams(selected_measure)

    if group_name is not None:
        suffix = group_name.title()
        measure_name = f"{measure.replace('_',' ').title()}"

        if multi_index == True:
            columns =  pd.MultiIndex.from_arrays([(suffix,suffix),(words_colname,measure_name )])
        else:
            columns = [f"{words_colname} ({suffix})", f"{measure_name} ({suffix})"]
    else:
        columns=[words_colname, measure.replace('_',' ').title()]
    # Convert the Scored n-grams to a dataframe
    df_ngrams = pd.DataFrame(scored_ngrams, columns=columns)

    # Clean up ngrams 
    if join_words == True:
        df_ngrams[columns[0]] = df_ngrams[columns[0]].map(sep.join)
    
    if top_n is not None:
        return df_ngrams.head(top_n)
    else:
        return df_ngrams



def compare_ngram_measures_df(group1_tokens, group2_tokens, ngrams=2,
                              measure='raw_freq', min_freq = 1, top_n=25,
                             words_colname='Words', group1_name=None, group2_name=None, 
                             multi_index=True):
    """Compare 2 groups ngrams side-by-side"""
    import pandas as pd
    
    group1_df = get_ngram_measures_df(group1_tokens,
                                      ngrams=ngrams,
                                      measure=measure,
                                      top_n=top_n,
                                      words_colname=words_colname,group_name=group1_name, multi_index=multi_index)
                                    
    group2_df = get_ngram_measures_df(group2_tokens,
                                      ngrams=ngrams,
                                      measure=measure,
                                      top_n=top_n,words_colname=words_colname,group_name=group2_name, multi_index=multi_index)  
    return pd.concat([group1_df, group2_df],axis=1)
    
    
    
def plot_group_ngrams( ngram_df, group1_colname = 'Low Ratings',
                      group2_colname= 'High Ratings',words_colname="Words",
                       plot_col="Raw Freq",top_n=None,
                      color_group1 = 'crimson', color_group2="green",
                      figsize=(12, 8),suptitle_kws={},suptitle_y= 1.01, rotation = 45):
    import matplotlib.pyplot as plt
    ### Plotting as Bar Graph
    if top_n == None:
        top_n = len(ngram_df)
        
    eda_ngrams_grp1 = ngram_df[group1_colname].set_index(words_colname)
    eda_ngrams_grp1 = eda_ngrams_grp1.sort_values(plot_col, ascending=True).tail(top_n)

    eda_ngrams_grp2 = ngram_df[group2_colname].set_index(words_colname)
    eda_ngrams_grp2 = eda_ngrams_grp2.sort_values(plot_col, ascending=True).tail(top_n)
                                                                            

    ## Plot the ngram frequencies
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    #
    eda_ngrams_grp1[plot_col].plot( kind="barh", 
                                   title=f"Top {top_n} n-grams for {group1_colname}",# (by {plot_col})", 
                                   ax=axes[0], 
                                   color=color_group1
                                  )
    eda_ngrams_grp2[plot_col].plot( kind="barh", 
                                   title=f"Top {top_n} n-grams for {group2_colname}",# (by {plot_col})", 
                                   ax=axes[1], 
                                   color=color_group2
                                  )
    # Set suptitle
    # determine which type of ngrams
    n_words = len(eda_ngrams_grp1.index[0].split(" "))
    if n_words == 2:
        ngrams = "Bigrams"
    elif n_words == 3:
        ngrams = "Trigrams"
    elif n_words == 4:
        ngrams = "Quadgrams"
    else:
        raise Exception(f"ngrams is {n_words}")
        
    
    fig.suptitle(f"Group {ngrams} by {plot_col}", y=suptitle_y, **suptitle_kws )
    for ax in axes:
        ax.spines["top"].set_visible(False)  # Remove the top spine
        ax.set_xlabel(plot_col) # Add the Measure label
        ax.spines["right"].set_visible(False)  # Remove the right spine
        
        # Fix label rotation
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')
    fig.tight_layout()
    return fig



## Get yearly data to plot
def get_rating_percent_by_year(df, melted=False, melted_value_name ="%", melted_var_name="Stars"):
    import pandas as pd
    plot_df = df.rename({'overall':'Rating Proportion'},axis=1)
    count_ratings_by_year  = plot_df.groupby('year',#as_index=False
                                       )['Rating Proportion'].value_counts(normalize=True).sort_index()
    
    count_ratings_by_year = count_ratings_by_year.unstack(1).sort_index().fillna(0)

    if melted==True:
        count_ratings_by_year = pd.melt(count_ratings_by_year.reset_index(drop=False),
                                        id_vars=['year'], 
                                        value_name=melted_value_name,
                                        var_name=melted_var_name)
    return count_ratings_by_year

def get_average_rating_by_year(df):
    avg_rating_by_year  = df.groupby('year',
                                )[['overall']].mean().sort_index()

    avg_rating_by_year = avg_rating_by_year.rename({'overall':"Average Rating"},axis=1)
    return avg_rating_by_year


import nltk
from wordcloud import STOPWORDS

def get_groups_freqs_wordclouds(df, ngrams=1, group_col='target-rating', text_col='review-text-full', 
                                as_freqs=True, as_tokens=False, tokenizer=nltk.casual_tokenize, 
                                drop_groups=[], stopwords=[*STOPWORDS]):
    """Get frequencies or raw texts for word clouds by group."""
    # Filter out unwanted groups upfront
    df_filtered = df[~df[group_col].isin(drop_groups)]
    
    # Make stopwords a set
    stopwords = set(stopwords)
    
    # Initialize result dictionary
    group_texts = {}
    
    # Process each group
    for group_name, group_df in df_filtered.groupby(group_col):
        # Handle list of tokens in text_col
        joined_texts = " ".join(group_df[text_col].explode().fillna('') if isinstance(group_df[text_col].iloc[0], list) else group_df[text_col])
        
        # Tokenize if necessary
        if as_tokens or as_freqs:
            tokens = [w.lower() for w in tokenizer(joined_texts) if w.lower() not in stopwords]
            
            # Generate n-grams if requested
            if ngrams > 1:
                tokens = [" ".join(ngram) for ngram in nltk.ngrams(tokens, ngrams)]
                
            if as_freqs:
                # Calculate frequency distribution
                group_texts[group_name] = dict(nltk.FreqDist(tokens))
            else:
                group_texts[group_name] = tokens
        else:
            group_texts[group_name] = joined_texts
            
    return group_texts



def make_wordclouds_from_freqs(groups_dict,  grp1_key="Low", grp2_key="High",
                               grp1_cmap = "Reds", grp2_cmap ="Greens",stopwords=None, 
                               width=800, height=1000, min_word_length=2, max_words=200,
                               min_font_size=6,
                               cloud_kws = {},
                               plot_clouds=True,
                               figsize=(8, 5), grp1_label=None, grp2_label=None, title=None,
                               title_params ={'y':1.01,"fontsize":'xx-large'}):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # Make word clouds of processed tokens
    shared_params = dict(random_state = 42,
                          width = width,
                          height = height, stopwords=stopwords,
                          min_word_length = min_word_length,
                          min_font_size=min_font_size,
                        **cloud_kws)

    # Slice groups from dict
    grp1_text = groups_dict[grp1_key]
    grp2_text = groups_dict[grp2_key]

    
    # Create an instance of a WordCloud but DO NOT GENERATE YET
    grp1_cloud = WordCloud(colormap=grp1_cmap, **shared_params)
    grp2_cloud = WordCloud(colormap=grp2_cmap, **shared_params)
    
    if isinstance(grp1_text, dict):
        grp1_cloud = grp1_cloud.generate_from_frequencies(grp1_text)
        grp2_cloud = grp2_cloud.generate_from_frequencies(grp2_text)
    else:
        grp1_cloud = grp1_cloud.generate(grp1_text)
        grp2_cloud = grp2_cloud.generate(grp2_text)

    if plot_clouds == False:
        return grp1_cloud, grp2_cloud
        
    else:
        # Use keys if no laebels given
        if grp1_label is None:
            grp1_label = grp1_key.title()
        if grp2_label is None: 
            grp2_label = grp2_key.title()
            
        return plot_wordclouds(grp1_cloud, grp2_cloud, grp1_cloud_label=grp1_label,figsize=figsize,
                             grp2_cloud_label=grp2_label,title=title, title_params=title_params)
         

def plot_wordclouds(grp1_cloud, grp2_cloud, 
                    grp1_cloud_label="Low Ratings", 
                    grp2_cloud_label='High Ratings',
                    title='Comparing Word Usage', figsize=(8, 5), 
                    title_params ={'y':1.0,"fontsize":'xx-large'},
                   ):
    """Plots the wordlcouds for two groups"""
    import matplotlib.pyplot as plt

    ## Plot the Images
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    
    axes[0].imshow(grp1_cloud)
    axes[0].set_title(grp1_cloud_label)
    axes[0].axis('off')

    axes[1].imshow(grp2_cloud)
    axes[1].set_title(grp2_cloud_label)
    axes[1].axis('off')

    if title is not None:
        fig.suptitle(title,**title_params)

    fig.tight_layout()    
    return fig


def get_stopwords_from_string(stopwords_to_add = None, default_stopwords=True, ):

    from nltk import casual_tokenize

    # Add nclude the 
    if default_stopwords == True:
        from wordcloud import STOPWORDS
        from string import punctuation
        custom_stopwords = [*STOPWORDS,*punctuation]
    else:
        custom_stopwords = []
        
    # If add_stopwords is a string, tokenize it first
    if isinstance(stopwords_to_add, str):
        add_stopwords = casual_tokenize(stopwords_to_add)
    else:
        add_stopwords = stopwords_to_add
    
    
    # Combine custom_stopwords
    combined_stopwords = [*custom_stopwords, *add_stopwords]
    return combined_stopwords