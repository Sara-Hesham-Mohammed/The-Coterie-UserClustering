from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import numpy as np
from Models.User_Model import User


class UserPreprocessor:
    def __init__(self):
        self.tag_encoder = MultiLabelBinarizer()
        self.country_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.language_encoder = MultiLabelBinarizer()
        self.size_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.fitted = False

    def fit(self, users):
        """Fit encoders on all users to create consistent feature space"""
        # Collect all unique values
        all_tags = [user.tags if user.tags else [] for user in users]
        all_countries = [[user.country] for user in users]
        all_languages = [user.languages if user.languages else [] for user in users]
        all_sizes = [[user.preferred_group_size] for user in users]

        # Fit encoders
        self.tag_encoder.fit(all_tags)
        self.country_encoder.fit(all_countries)
        self.language_encoder.fit(all_languages)
        self.size_encoder.fit(all_sizes)

        self.fitted = True
        return self

    def transform(self, user):
        """Transform a single user's features"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transforming")

        encoded_features = []

        # Encode tags (multi-label)
        if user.tags:
            tag_encoded = self.tag_encoder.transform([user.tags])
        else:
            tag_encoded = self.tag_encoder.transform([[]])
        encoded_features.append(tag_encoded)

        # Encode country (single label)
        country_encoded = self.country_encoder.transform([[user.country]])
        encoded_features.append(country_encoded)

        # Encode languages (multi-label)
        if user.languages:
            lang_encoded = self.language_encoder.transform([user.languages])
        else:
            lang_encoded = self.language_encoder.transform([[]])
        encoded_features.append(lang_encoded)

        # Encode preferred group size (single label)
        size_encoded = self.size_encoder.transform([[user.preferred_group_size]])
        encoded_features.append(size_encoded)

        # Combine all features
        combined_features = np.hstack(encoded_features)
        return combined_features

    def fit_transform(self, users):
        """Fit on all users and return transformed features for all users"""
        self.fit(users)
        return np.vstack([self.transform(user) for user in users])

    def get_feature_names(self):
        """Get feature names for interpretability"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")

        feature_names = []

        # Tag feature names
        tag_names = [f"tag_{tag}" for tag in self.tag_encoder.classes_]
        feature_names.extend(tag_names)

        # Country feature names
        country_names = [f"country_{country[0]}" for country in self.country_encoder.categories_[0]]
        feature_names.extend(country_names)

        # Language feature names
        lang_names = [f"language_{lang}" for lang in self.language_encoder.classes_]
        feature_names.extend(lang_names)

        # Size feature names
        size_names = [f"size_{size[0]}" for size in self.size_encoder.categories_[0]]
        feature_names.extend(size_names)

        return feature_names


# Legacy function for backward compatibility
def preproc(users):
    """
    Preprocesses multiple users for clustering by encoding categorical features.
    Returns the feature matrix for all users.
    """
    if isinstance(users, User):
        users = [users]

    preprocessor = UserPreprocessor()
    return preprocessor.fit_transform(users)

