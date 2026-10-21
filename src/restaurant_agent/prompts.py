"""Prompts for the steps that use the LLM."""

EXTRACT_PROMPT = """\
You extract restaurant search preferences from a user's request.

Only fill a field if the user stated it. Otherwise leave it null; never guess.

- location: the area, landmark or city to search in, as the user wrote it.
  Null for "near me", "nearby", or when no place is mentioned.
- cuisine: the cuisine or dish style, e.g. "South Indian", "biryani".
- budget: maximum spend PER PERSON in Indian rupees, as a whole number.
  If a total is given for a group ("₹1500 for two"), divide by the group size.
  Vague words like "cheap" or "affordable" without a number -> null.
- min_rating: minimum Google rating on a 0-5 scale.
  "rating > 4.2", "4.2+", "at least 4.2" -> 4.2. "highly rated" without a number -> null.
- diet: a dietary need for this search, e.g. "vegetarian", "vegan", "Jain", "halal".

remember: facts about the user that hold beyond this one search, e.g. "I'm
vegetarian", "I live in Gunjur, Bangalore", "remember my budget is ₹500", "I only
want places rated 4+". Put a home or usual area in home_location. A one-off
detail ("tonight", "this time", or just part of the request) is NOT remembered.
A remembered fact that applies to this search also goes in the search fields
(e.g. "I'm vegetarian, find dosa" -> diet "vegetarian" and remember.diet "vegetarian").

Follow-ups: if a "Previous search" is given, decide whether the new message
changes that search ("something cheaper", "what about Koramangala?", "make it
4.5+") or starts a new one ("find me biryani in Whitefield").
- Change: start from the previous values and change only what the message
  changes. "Cheaper" without a number -> previous budget minus about a third,
  rounded to a multiple of 50 (null if there was no previous budget).
- New search: ignore the previous values.
"""

RANK_PROMPT = """\
You pick the best restaurants for a user's request from a numbered list of candidates.

Every candidate already meets the user's hard requirements (open, minimum rating,
budget), but the search can also return places that don't serve what the user asked
for. The list is sorted by a score based on rating, number of reviews and distance.

1. Drop candidates that don't match the requested cuisine or dish. Judge by the type
   in brackets, the name and what diners say: keep a candidate only if they make a
   match likely. For example, drop a "North Indian Restaurant" for a South Indian
   request, or a generic "Restaurant" with an unrelated name and no matching reviews
   for an Ethiopian request. A generic type is fine when the name or reviews show a
   match (e.g. reviews praising Madurai-style parotta for a South Indian request). If the request names
   no cuisine or dish, every candidate matches. If a dietary need is given, also drop
   candidates whose type or name clearly conflicts with it (e.g. a "Seafood
   Restaurant" for a vegetarian).
2. From the rest, pick up to {top_n}, best first. Keep the list order unless the
   request gives a reason to differ, e.g. a specific dish, style or occasion that a
   restaurant's type, name or reviews match better.
3. If no candidate matches, answer with an empty picks list ("picks": []).
   Fewer good picks are better than unrelated ones.

For each pick, write one short sentence on why it fits, using only facts given in
the list (type, rating, reviews, price, distance, name, address, what diners say).
Mention a dish or quality only if "Diners say" mentions it. Do not invent dishes,
ambience or anything else that isn't listed.

Always answer through the structured output, never in plain text. Refer to
candidates by their numbers, not their names.
"""
