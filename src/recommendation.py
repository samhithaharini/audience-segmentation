# src/recommendation.py
"""
Dynamic recommendation engine mapping audience segment profiles to personalized OTT content.
"""

from typing import Dict, List, Any

# Curated catalog by genre and engagement tier
GENRE_CATALOG = {
    'Action': ['Extraction 2', 'John Wick 4', 'Avengers: Endgame', 'Mad Max: Fury Road', 'Top Gun: Maverick', 'The Dark Knight'],
    'Comedy': ['Brooklyn Nine-Nine', 'The Office', 'Ted Lasso', 'Superbad', 'Parks and Recreation', 'Schitt\'s Creek'],
    'Drama': ['Succession', 'The Crown', 'Breaking Bad', 'Ozark', 'Peaky Blinders', 'Better Call Saul'],
    'Sci-Fi': ['Interstellar', 'Dune: Part Two', 'Stranger Things', 'Dark', 'The Expanse', 'Black Mirror'],
    'Documentary': ['Our Planet', 'The Last Dance', 'Formula 1: Drive to Survive', 'The Social Dilemma', '13th', 'Free Solo'],
    'Romance': ['The Notebook', 'Bridgerton', 'Pride & Prejudice', 'Past Lives', 'La La Land', 'About Time'],
    'Horror': ['The Conjuring', 'Hereditary', 'A Quiet Place', 'Get Out', 'Talk to Me', 'Smile'],
    'Thriller': ['Severance', 'Mindhunter', 'Shutter Island', 'Squid Game', 'Gone Girl', 'Prisoners'],
    'Animation': ['Spider-Man: Across the Spider-Verse', 'Arcane', 'Spirited Away', 'Inside Out 2', 'Toy Story 4']
}

DEFAULT_TOP_PICKS = ['Stranger Things', 'Breaking Bad', 'Inception', 'The Crown', 'Dune', 'Succession']

def get_recommendations(cluster_name: str, profile_stats: Dict[str, Any] = None) -> List[str]:
    """
    Return personalized OTT titles based on cluster name and profile characteristics.
    """
    cluster_lower = cluster_name.lower()

    # 1. Match by primary genre keyword in name
    for genre, titles in GENRE_CATALOG.items():
        if genre.lower() in cluster_lower:
            return titles

    # 2. Match by profile mode_Favorite_Genre if available
    if profile_stats and 'categorical_modes' in profile_stats:
        modes = profile_stats['categorical_modes']
        for col, val in modes.items():
            if 'genre' in col.lower() and str(val) in GENRE_CATALOG:
                return GENRE_CATALOG[str(val)]

    # 3. Match by engagement keywords
    if 'binge' in cluster_lower or 'at-risk' in cluster_lower:
        return ['Breaking Bad', 'Stranger Things', 'Squid Game', 'Succession', 'Dark', 'Ozark']
    elif 'casual' in cluster_lower:
        return ['The Office', 'Brooklyn Nine-Nine', 'Ted Lasso', 'Modern Family', 'Parks and Recreation']

    return DEFAULT_TOP_PICKS