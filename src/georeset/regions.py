"""Geofabrik extract names to clean region names and continents."""

_CONTINENTS: dict[str, str] = {
    "africa": (
        "algeria angola benin botswana burkina-faso burundi cameroon canary-islands cape-verde "
        "central-african-republic chad comores congo-brazzaville congo-democratic-republic "
        "djibouti egypt equatorial-guinea eritrea ethiopia gabon ghana guinea guinea-bissau "
        "ivory-coast kenya lesotho liberia libya madagascar malawi mali mauritania mauritius "
        "mayotte morocco mozambique namibia niger nigeria reunion rwanda "
        "saint-helena-ascension-and-tristan-da-cunha sao-tome-and-principe senegal-and-gambia "
        "seychelles sierra-leone somalia south-africa south-sudan sudan swaziland tanzania togo "
        "tunisia uganda zambia zimbabwe"
    ),
    "antarctica": "antarctica",
    "asia": (
        "afghanistan armenia azerbaijan bangladesh bhutan cambodia east-timor gcc-states iran "
        "iraq israel-and-palestine jordan kazakhstan kyrgyzstan laos lebanon "
        "malaysia-singapore-brunei maldives mongolia myanmar nepal north-korea pakistan "
        "philippines south-korea sri-lanka syria taiwan tajikistan thailand turkmenistan "
        "uzbekistan vietnam yemen russia-far-eastern-fed-district russia-siberian-fed-district "
        "russia-ural-fed-district china-anhui china-beijing china-chongqing china-fujian "
        "china-gansu china-guangdong china-guangxi china-guizhou china-hainan china-hebei "
        "china-heilongjiang china-henan china-hong-kong china-hubei china-hunan "
        "china-inner-mongolia china-jiangsu china-jiangxi china-jilin china-liaoning "
        "china-macau china-ningxia china-qinghai china-shaanxi china-shandong china-shanghai "
        "china-shanxi china-sichuan china-tianjin china-tibet china-xinjiang china-yunnan "
        "china-zhejiang india-central-zone india-eastern-zone india-north-eastern-zone "
        "india-northern-zone india-southern-zone india-western-zone indonesia-java "
        "indonesia-kalimantan indonesia-maluku indonesia-nusa-tenggara indonesia-sulawesi "
        "indonesia-sumatra japan-chubu japan-chugoku japan-hokkaido japan-kansai japan-kanto "
        "japan-kyushu japan-shikoku japan-tohoku"
    ),
    "europe": (
        "albania alsace andalucia andorra aquitaine aragon asturias austria auvergne azores "
        "baden-wuerttemberg basse-normandie bayern belarus belgium berlin bosnia-herzegovina "
        "bourgogne brandenburg bremen bretagne bulgaria cantabria castilla-la-mancha "
        "castilla-y-leon cataluna centre centro ceuta champagne-ardenne corse croatia cyprus "
        "czech-republic denmark england estonia extremadura faroe-islands finland "
        "franche-comte galicia georgia greece guernsey-jersey hamburg haute-normandie hessen "
        "hungary iceland ile-de-france ireland-and-northern-ireland islas-baleares "
        "isle-of-man italy kosovo la-rioja languedoc-roussillon latvia liechtenstein limousin "
        "lithuania lorraine luxembourg macedonia madrid malta mecklenburg-vorpommern melilla "
        "midi-pyrenees moldova monaco montenegro murcia navarra netherlands niedersachsen "
        "nord-est nord-pas-de-calais nordrhein-westfalen norway pais-vasco pays-de-la-loire "
        "picardie poitou-charentes poland portugal provence-alpes-cote-d-azur "
        "rheinland-pfalz rhone-alpes romania russia-central-fed-district "
        "russia-crimean-fed-district russia-kaliningrad russia-north-caucasus-fed-district "
        "russia-northwestern-fed-district russia-south-fed-district russia-volga-fed-district "
        "saarland sachsen sachsen-anhalt schleswig-holstein scotland serbia slovakia slovenia "
        "sud sweden switzerland thueringen turkey ukraine valencia wales"
    ),
    "north-america": (
        "bermuda canada-alberta canada-british-columbia canada-manitoba canada-new-brunswick "
        "canada-newfoundland-and-labrador canada-northwest-territories canada-nova-scotia "
        "canada-nunavut canada-ontario canada-prince-edward-island canada-quebec "
        "canada-saskatchewan canada-yukon greenland mexico us-alabama us-alaska us-arizona "
        "us-arkansas us-california us-colorado us-connecticut us-delaware "
        "us-district-of-columbia us-florida us-georgia us-hawaii us-idaho us-illinois "
        "us-indiana us-iowa us-kansas us-kentucky us-louisiana us-maine us-maryland "
        "us-massachusetts us-michigan us-minnesota us-mississippi us-missouri us-montana "
        "us-nebraska us-nevada us-new-hampshire us-new-jersey us-new-mexico us-new-york "
        "us-north-carolina us-north-dakota us-ohio us-oklahoma us-oregon us-pennsylvania "
        "us-rhode-island us-south-carolina us-south-dakota us-tennessee us-texas us-utah "
        "us-vermont us-virginia us-washington us-west-virginia us-wisconsin us-wyoming"
    ),
    "central-america": (
        "bahamas belize costa-rica cuba el-salvador guadeloupe guatemala haiti-and-domrep "
        "honduras jamaica martinique nicaragua panama us-puerto-rico us-virgin-islands"
    ),
    "south-america": (
        "argentina bolivia brazil-centro-oeste brazil-nordeste brazil-norte brazil-sudeste "
        "brazil-sul chile colombia ecuador falklands guyana guyane paraguay peru suriname "
        "uruguay venezuela"
    ),
    "oceania": (
        "american-oceania australia cook-islands fiji ile-de-clipperton indonesia-papua "
        "kiribati marshall-islands micronesia nauru new-caledonia new-zealand niue palau "
        "papua-new-guinea pitcairn-islands polynesie-francaise samoa solomon-islands tokelau "
        "tonga tuvalu vanuatu wallis-et-futuna"
    ),
}

CONTINENT_OF: dict[str, str] = {
    region: continent
    for continent, names in _CONTINENTS.items()
    for region in names.split()
}


SUFFIX = "-latest"


def region_name(stem: str) -> str:
    """Map a Geofabrik file stem (``italy-latest``) to a region name (``italy``)."""
    return stem.removesuffix(SUFFIX)
