"""Typed data models passed between the steps."""

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """Standing preferences saved per user and applied to later searches."""

    home_location: str | None = Field(
        default=None,
        description="Where the user lives or usually eats, e.g. 'Gunjur, Bengaluru'. "
        "Used for 'near me' searches.",
    )
    diet: str | None = Field(
        default=None, description="Dietary need, e.g. 'vegetarian', 'vegan', 'Jain', 'halal'."
    )
    budget: int | None = Field(
        default=None, gt=0, description="Usual maximum spend per person in Indian rupees."
    )
    min_rating: float | None = Field(
        default=None, ge=0, le=5, description="Usual minimum Google rating, 0-5."
    )


class Preferences(BaseModel):
    """Restaurant search preferences stated in the user's request. Null = not stated."""

    location: str | None = Field(
        default=None,
        description="Area, landmark or city to search in, as written. "
        "Null for 'near me', 'nearby' or no location.",
    )
    cuisine: str | None = Field(
        default=None, description="Cuisine or dish style, e.g. 'South Indian', 'biryani'."
    )
    budget: int | None = Field(
        default=None, gt=0, description="Maximum spend per person in Indian rupees."
    )
    min_rating: float | None = Field(
        default=None, ge=0, le=5, description="Minimum Google rating, 0-5."
    )
    diet: str | None = Field(
        default=None, description="Dietary need for this search, e.g. 'vegetarian'."
    )
    remember: UserPreferences = Field(
        default_factory=UserPreferences,
        description="Facts about the user that hold beyond this search. Empty unless stated.",
    )


class Pick(BaseModel):
    number: int = Field(description="The candidate's number from the list.")
    reason: str = Field(description="One sentence on why it fits, using only the listed facts.")


class LLMRanking(BaseModel):
    picks: list[Pick] = Field(description="Best match first.")
