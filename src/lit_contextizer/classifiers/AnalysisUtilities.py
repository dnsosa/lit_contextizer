"""Collection of functions for analyzing results."""

# -*- coding: utf-8 -*-

from lit_contextizer.data_models.Utilities import drop_the_s


def include_annotations_in_insider_corpus(insider_df, features_df):
    """
    Add the annotation feature to the input dataframe of features made from insider corpus.

    :param insider_df: insider DF before extracted features
    :param features_Df: insider DF with features extracted
    """
    insider_positives = list(zip(list(insider_df['Extracted Relation']), list(insider_df['Context'])))

    def declare_annotation(row, positives=insider_positives):
        positives_plural = [(rel, con + 's') for rel, con in positives]
        all_positives = positives + positives_plural
        return (row['rel'], row['con']) in set(all_positives)

    features_df_copy = features_df.copy()
    features_df_copy['annotation'] = features_df_copy.apply(declare_annotation, axis=1)

    def drop_the_s_con(row):
        return drop_the_s(row['con'])

    features_df_copy['con'] = features_df_copy.apply(drop_the_s_con, axis=1)

    # Hide the insider sentence as the label!
    annotated_features_df = features_df_copy[~(features_df_copy.annotation & (features_df_copy.sent_dist == 0))].\
        drop_duplicates()

    # Remove relations that are not insider sentences
    annotated_features_df = annotated_features_df.merge(insider_df,
                                                        left_on="rel",
                                                        right_on="Extracted Relation",
                                                        how="inner")

    # Finally, recalculate "is_closest", which is facilitated by Pandas DF groupby
    df_grp = annotated_features_df.groupby(['rel'])['sent_dist']
    annotated_features_df = annotated_features_df.assign(min_sent_dist=df_grp.transform(min))
    t1 = annotated_features_df['sent_dist']
    t2 = annotated_features_df['min_sent_dist']
    annotated_features_df["is_closest_cont_by_sent"] = (t1 == t2)

    return annotated_features_df
