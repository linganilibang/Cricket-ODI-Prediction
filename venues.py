"""Clean up Cricsheet venue names and work out which country hosts each ground.

Cricsheet writes the same ground several ways ("Lord's" and "Lord's, London"),
and it has no country field. Two helpers fix that:

    canonical_venue(venue)  one name per ground
    host_team(venue, city)  the team whose home country the ground is in,
                            using the same names as the match data
                            (e.g. "West Indies" for any Caribbean ground)

The country list is written by hand from the ground and city names. Check it
with `python venues.py --raw path/to/odis_json`, which lists anything unmatched.
"""
import re

# Renamed or differently spelled grounds that the "strip the city" rule cannot join.
ALIASES = {
    "feroz shah kotla": "arun jaitley stadium",
    "sardar patel (gujarat) stadium": "narendra modi stadium",
    "sardar patel stadium": "narendra modi stadium",
    "brisbane cricket ground": "the gabba",
    "westpac stadium": "sky stadium",
    "westpac park": "seddon park",
    "jade stadium": "ami stadium",
    "western australia cricket association ground": "w.a.c.a. ground",
    "goodyear park": "mangaung oval",
    "outsurance oval": "mangaung oval",
    "diamond oval": "de beers diamond oval",
    "boland bank park": "boland park",
    "daren sammy national cricket stadium": "darren sammy national cricket stadium",
    "zohur ahmed chowdhury stadium": "zahur ahmed chowdhury stadium",
    "sher-e-bangla national cricket stadium": "shere bangla national stadium",
    "r.premadasa stadium": "r premadasa stadium",
    "m.chinnaswamy stadium": "m chinnaswamy stadium",
    "sheikh zayed stadium": "zayed cricket stadium",
    "sharjah cricket association stadium": "sharjah cricket stadium",
    "dubai sports city cricket stadium": "dubai international cricket stadium",
    "icc global cricket academy": "icc academy",
    "vra cricket ground": "vra ground",
    "grange cricket club": "grange cricket club ground",
    "vidarbha c.a. ground": "vidarbha cricket association stadium",
    "vidarbha cricket association ground": "vidarbha cricket association stadium",
    "punjab cricket association is bindra stadium": "punjab cricket association stadium",
    "niranjan shah stadium": "saurashtra cricket association stadium",
    "madhavrao scindia cricket ground": "saurashtra cricket association stadium",
    "the wanderers stadium": "new wanderers stadium",
    "wanderers stadium": "new wanderers stadium",
    "wanderers cricket ground": "wanderers cricket ground",
    "moosa cricket stadium": "moosa cricket stadium",
    "choice moosa stadium": "moosa cricket stadium",
    "the oval": "kennington oval",
    "mission road ground": "mission road ground",
    "sedgars park": "senwes park",
    "arnos vale ground": "arnos vale ground",
    "queen's park oval": "queen's park oval",
    "kensington oval": "kensington oval",
    "sabina park": "sabina park",
    "providence stadium": "providence stadium",
    "sir vivian richards stadium": "sir vivian richards stadium",
    "warner park": "warner park",
    "tribhuvan university international cricket ground": "tribhuvan university international cricket ground",
    "dr. y.s. rajasekhara reddy aca-vdca cricket stadium": "dr. y.s. rajasekhara reddy aca-vdca cricket stadium",
    "the rose bowl": "the rose bowl",
    "the royal & sun alliance county ground": "county ground bristol",
    "the village": "the village, malahide",
    "malahide": "the village, malahide",
}

# Host country by city or ground keyword. First match wins, so longer or more
# specific keywords come first. Values are team names as Cricsheet writes them.
_COUNTRY_KEYWORDS = [
    # West Indies (every Caribbean island plays as the West Indies team)
    ("west indies", "West Indies"),
    ("barbados", "West Indies"), ("bridgetown", "West Indies"), ("kensington oval", "West Indies"),
    ("antigua", "West Indies"), ("north sound", "West Indies"), ("sir vivian richards", "West Indies"),
    ("antigua recreation", "West Indies"), ("st lucia", "West Indies"), ("gros islet", "West Indies"),
    ("beausejour", "West Indies"), ("darren sammy", "West Indies"), ("daren sammy", "West Indies"),
    ("st vincent", "West Indies"), ("kingstown", "West Indies"), ("arnos vale", "West Indies"),
    ("grenada", "West Indies"), ("| st george's", "West Indies"),
    ("queen's park (new)", "West Indies"), ("trinidad", "West Indies"), ("port of spain", "West Indies"),
    ("tarouba", "West Indies"), ("queen's park oval", "West Indies"), ("brian lara", "West Indies"),
    ("jamaica", "West Indies"), ("kingston", "West Indies"), ("sabina park", "West Indies"),
    ("guyana", "West Indies"), ("providence", "West Indies"),
    ("st kitts", "West Indies"), ("basseterre", "West Indies"), ("warner park", "West Indies"),
    ("dominica", "West Indies"), ("roseau", "West Indies"),
    # United States and Canada
    ("lauderhill", "United States of America"), ("central broward", "United States of America"),
    ("dallas", "United States of America"), ("grand prairie", "United States of America"),
    ("pearland", "United States of America"), ("moosa", "United States of America"),
    ("king city", "Canada"), ("maple leaf", "Canada"), ("toronto", "Canada"),
    # India
    ("visakhapatnam", "India"), ("delhi", "India"), ("arun jaitley", "India"), ("feroz shah", "India"),
    ("cuttack", "India"), ("guwahati", "India"), ("mumbai", "India"), ("wankhede", "India"), ("brabourne", "India"),
    ("lucknow", "India"), ("gwalior", "India"), ("kolkata", "India"), ("eden gardens", "India"),
    ("kanpur", "India"), ("thiruvananthapuram", "India"), ("greenfield", "India"), ("dharamsala", "India"),
    ("indore", "India"), ("holkar", "India"), ("vadodara", "India"), ("baroda", "India"),
    ("ranchi", "India"), ("jamshedpur", "India"), ("bangalore", "India"), ("bengaluru", "India"),
    ("chinnaswamy", "India"), ("chennai", "India"), ("chepauk", "India"), ("rajkot", "India"),
    ("chandigarh", "India"), ("mohali", "India"), ("new chandigarh", "India"), ("pune", "India"),
    ("poona", "India"), ("maharashtra cricket", "India"), ("hyderabad, deccan", "India"),
    ("uppal", "India"), ("rajiv gandhi international", "India"), ("lal bahadur", "India"),
    ("ahmedabad", "India"), ("motera", "India"), ("narendra modi", "India"), ("jaipur", "India"),
    ("sawai mansingh", "India"), ("nagpur", "India"), ("vidarbha", "India"), ("raipur", "India"),
    ("faridabad", "India"), ("margao", "India"), ("fatorda", "India"), ("nehru stadium", "India"),
    ("green park", "India"), ("barsapara", "India"),
    # Pakistan (Niaz Stadium Hyderabad is in Sindh)
    ("peshawar", "Pakistan"), ("lahore", "Pakistan"), ("gaddafi", "Pakistan"), ("faisalabad", "Pakistan"),
    ("iqbal stadium", "Pakistan"), ("karachi", "Pakistan"), ("rawalpindi", "Pakistan"), ("multan", "Pakistan"),
    ("sheikhupura", "Pakistan"), ("niaz stadium", "Pakistan"), ("| sind", "Pakistan"),
    # Bangladesh
    ("dhaka", "Bangladesh"), ("bangabandhu", "Bangladesh"), ("chattogram", "Bangladesh"), ("chittagong", "Bangladesh"),
    ("zahur ahmed", "Bangladesh"), ("zohur ahmed", "Bangladesh"), ("ma aziz", "Bangladesh"), ("matiur", "Bangladesh"),
    ("fatullah", "Bangladesh"), ("khan shaheb", "Bangladesh"), ("osmani", "Bangladesh"), ("mirpur", "Bangladesh"),
    ("shere bangla", "Bangladesh"), ("sher-e-bangla", "Bangladesh"), ("bogra", "Bangladesh"), ("khulna", "Bangladesh"),
    ("sylhet", "Bangladesh"),
    # Sri Lanka
    ("colombo", "Sri Lanka"), ("premadasa", "Sri Lanka"), ("sinhalese", "Sri Lanka"), ("saravanamuttu", "Sri Lanka"),
    ("hambantota", "Sri Lanka"), ("mahinda rajapaksa", "Sri Lanka"), ("galle", "Sri Lanka"),
    ("pallekele", "Sri Lanka"), ("kandy", "Sri Lanka"), ("dambulla", "Sri Lanka"),
    # Nepal
    ("kathmandu", "Nepal"), ("kirtipur", "Nepal"), ("tribhuvan", "Nepal"), ("mulpani", "Nepal"),
    # Australia
    ("adelaide", "Australia"), ("hobart", "Australia"), ("bellerive", "Australia"), ("brisbane", "Australia"),
    ("woolloongabba", "Australia"), ("cairns", "Australia"), ("cazaly", "Australia"), ("bundaberg", "Australia"),
    ("melbourne", "Australia"), ("docklands", "Australia"), ("mackay", "Australia"), ("great barrier reef", "Australia"),
    ("canberra", "Australia"), ("manuka", "Australia"), ("darwin", "Australia"), ("marrara", "Australia"),
    ("perth", "Australia"), ("w.a.c.a", "Australia"), ("western australia", "Australia"), ("townsville", "Australia"),
    ("riverway", "Australia"), ("tony ireland", "Australia"), ("sydney", "Australia"),
    # New Zealand
    ("ami stadium", "New Zealand"), ("christchurch", "New Zealand"), ("hagley", "New Zealand"),
    ("jade stadium", "New Zealand"), ("lincoln", "New Zealand"), ("bert sutcliffe", "New Zealand"),
    ("wellington", "New Zealand"), ("basin reserve", "New Zealand"), ("westpac", "New Zealand"),
    ("sky stadium", "New Zealand"), ("mount maunganui", "New Zealand"), ("bay oval", "New Zealand"),
    ("dunedin", "New Zealand"), ("carisbrook", "New Zealand"), ("university oval", "New Zealand"),
    ("whangarei", "New Zealand"), ("cobham", "New Zealand"), ("auckland", "New Zealand"), ("eden park", "New Zealand"),
    ("hamilton", "New Zealand"), ("seddon park", "New Zealand"), ("napier", "New Zealand"), ("mclean park", "New Zealand"),
    ("nelson", "New Zealand"), ("saxton", "New Zealand"), ("queenstown", "New Zealand"),
    ("john davies oval", "New Zealand"), ("davies park", "New Zealand"),
    # South Africa
    ("paarl", "South Africa"), ("boland", "South Africa"), ("east london", "South Africa"), ("buffalo park", "South Africa"),
    ("bloemfontein", "South Africa"), ("mangaung", "South Africa"), ("goodyear", "South Africa"),
    ("outsurance", "South Africa"), ("chevrolet park", "South Africa"), ("pietermaritzburg", "South Africa"),
    ("kimberley", "South Africa"), ("diamond oval", "South Africa"), ("durban", "South Africa"),
    ("kingsmead", "South Africa"), ("johannesburg", "South Africa"), ("wanderers", "South Africa"),
    ("cape town", "South Africa"), ("newlands", "South Africa"), ("potchefstroom", "South Africa"),
    ("senwes", "South Africa"), ("sedgars", "South Africa"), ("port elizabeth", "South Africa"),
    ("gqeberha", "South Africa"), ("st george's park", "South Africa"), ("centurion", "South Africa"),
    ("supersport park", "South Africa"), ("benoni", "South Africa"), ("willowmoore", "South Africa"),
    # Zimbabwe
    ("harare", "Zimbabwe"), ("bulawayo", "Zimbabwe"), ("queens sports club", "Zimbabwe"),
    ("old hararians", "Zimbabwe"), ("takashinga", "Zimbabwe"),
    # England and Wales (one team) and Scotland, Ireland
    ("bristol", "England"), ("county ground", "England"), ("chelmsford", "England"), ("birmingham", "England"),
    ("edgbaston", "England"), ("leeds", "England"), ("headingley", "England"), ("london", "England"),
    ("lord's", "England"), ("kennington", "England"), ("manchester", "England"), ("old trafford", "England"),
    ("chester-le-street", "England"), ("riverside", "England"), ("cardiff", "England"), ("sophia gardens", "England"),
    ("canterbury", "England"), ("st lawrence", "England"), ("southampton", "England"), ("rose bowl", "England"),
    ("taunton", "England"), ("nottingham", "England"), ("trent bridge", "England"), ("the oval", "England"),
    ("dundee", "Scotland"), ("forthill", "Scotland"), ("| ayr", "Scotland"), ("cambusdoon", "Scotland"),
    ("edinburgh", "Scotland"), ("grange cricket", "Scotland"), ("aberdeen", "Scotland"), ("mannofield", "Scotland"),
    ("glasgow", "Scotland"), ("titwood", "Scotland"),
    ("dublin", "Ireland"), ("castle avenue", "Ireland"), ("clontarf", "Ireland"), ("malahide", "Ireland"),
    ("belfast", "Ireland"), ("stormont", "Ireland"), ("civil service cricket", "Ireland"),
    ("bready", "Ireland"), ("magheramason", "Ireland"),
    # Netherlands
    ("rotterdam", "Netherlands"), ("hazelaarweg", "Netherlands"), ("deventer", "Netherlands"), ("schootsveld", "Netherlands"),
    ("utrecht", "Netherlands"), ("maarschalkerweerd", "Netherlands"), ("the hague", "Netherlands"), ("westvliet", "Netherlands"),
    ("amstelveen", "Netherlands"), ("vra ", "Netherlands"),
    # Namibia, Kenya, Papua New Guinea, Hong Kong, Malaysia
    ("windhoek", "Namibia"), ("affies", "Namibia"), ("namibia", "Namibia"),
    ("nairobi", "Kenya"), ("mombasa", "Kenya"), ("gymkhana club ground", "Kenya"),
    ("port moresby", "Papua New Guinea"), ("amini park", "Papua New Guinea"),
    ("hong kong", "Hong Kong"), ("mong kok", "Hong Kong"),
    ("kuala lumpur", "Malaysia"), ("bangi", "Malaysia"), ("bayuemas", "Malaysia"), ("kinrara", "Malaysia"),
    # UAE and Oman
    ("dubai", "United Arab Emirates"), ("icc academy", "United Arab Emirates"), ("sharjah", "United Arab Emirates"),
    ("abu dhabi", "United Arab Emirates"), ("zayed", "United Arab Emirates"),
    ("al amarat", "Oman"), ("al amerat", "Oman"), ("oman", "Oman"),
]


def canonical_venue(venue):
    """One name per ground: drop the ", City" tail, then join known renames."""
    v = (venue or "unknown").strip().lower()
    if v in ALIASES:
        return ALIASES[v]
    # Names that really contain a comma are listed as aliases above; otherwise the
    # text before the first comma is the ground and the rest is the place.
    base = re.split(r",", v, maxsplit=1)[0].strip()
    base = ALIASES.get(base, base)
    return base


def host_team(venue, city):
    """Team name of the country that hosts the ground, or "" if not recognised."""
    text = f"{(venue or '').lower()} | {(city or '').lower()}"
    for key, team in _COUNTRY_KEYWORDS:
        if key in text:
            return team
    return ""


if __name__ == "__main__":
    import argparse, collections, glob, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    seen = {}
    for p in glob.glob(os.path.join(args.raw, "*.json")):
        info = json.load(open(p))["info"]
        if info.get("gender") == "male":
            seen[info.get("venue")] = info.get("city")
    miss = [(v, c) for v, c in seen.items() if not host_team(v, c)]
    print(len(seen), "venues,", len({canonical_venue(v) for v in seen}), "grounds after cleaning")
    print("unmatched:", miss)
    by = collections.defaultdict(set)
    for v, c in seen.items():
        by[canonical_venue(v)].add(host_team(v, c))
    print("grounds with two different countries:", {k: v for k, v in by.items() if len(v) > 1})
    for team, n in collections.Counter(host_team(v, c) for v, c in seen.items()).most_common():
        print(f"{n:4d} {team}")
