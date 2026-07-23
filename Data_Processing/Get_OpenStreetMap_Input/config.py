Continent_list = ['africa', 'usa', 'canada', 'italy', 'germany', 'netherlands', 'russia', 'spain', 'united-kingdom', 'france', 'poland', 
                  'asia', 'australia-oceania', 'central-america', 'europe', 'north-america', 'south-america']

#africa: 54 | 54
#usa: 52 | 51 (no puerto-rico hawaii)
#canada: 13 | 7 
#italy: 5 | 5 
#germany: 29 | 29 
#netherlands: 12 | 12 
#russia: 10 | 4 (central-fed-district far-eastern-fed-district south-fed-district volga-fed-district)
#spain: 18 | 18 
#united-kingdom: 3 | 3 
#france: 23 | 23 
#poland: 16 | 16 
#asia: 98 | 94 
#australia-oceania: 23 | 17
#central-america: 10 | 10
#europe: 42 | 42
#north-america: 2 | 2 
#south-america: 16 | 16

Region_list = {
    'africa': ['algeria', 'libya', 'angola', 'madagascar', 'benin', 'malawi', 'botswana', 'mali', 'burkina-faso', 'mauritania', 'burundi', 'mauritius',
               'cameroon', 'morocco', 'canary-islands', 'mozambique', 'cape-verde', 'namibia', 'central-african-republic', 'nigeria', 'chad', 'niger', 'comores', 'rwanda',
               'congo-brazzaville', 'saint-helena-ascension-and-tristan-da-cunha', 'congo-democratic-republic', 'sao-tome-and-principe', 'djibouti',
               'senegal-and-gambia', 'egypt', 'seychelles', 'equatorial-guinea', 'sierra-leone', 'eritrea', 'somalia', 'ethiopia', 'south-africa',
               'gabon', 'south-sudan', 'ghana', 'sudan', 'guinea-bissau', 'swaziland', 'guinea', 'tanzania', 'ivory-coast', 'togo', 'kenya', 'tunisia',
               'lesotho', 'uganda','liberia','zimbabwe'],
    'usa': ['alabama','alaska','arizona','arkansas','colorado','connecticut','delaware','district-of-columbia','florida','georgia','hawaii',
            'idaho','illinois','indiana','iowa','kansas','kentucky','louisiana','maine','maryland','massachusetts','michigan',
            'minnesota','mississippi','missouri','montana','nebraska','nevada','new-hampshire','new-jersey','new-mexico','new-york','norcal',
            'north-carolina','north-dakota','ohio','oklahoma','oregon','pennsylvania','puerto-rico','rhode-island','socal','south-carolina','south-dakota',
            'tennessee','texas','utah','vermont','virginia','washington','west-virginia','wisconsin','wyoming'],
    'canada': ['alberta','british-columbia','manitoba','new-brunswick','newfoundland-and-labrador','northwest-territories',
                'nova-scotia','nunavut','ontario','prince-edward-island','quebec','saskatchewan','yukon'],
    'italy': ['centro', 'nord-est', 'sud', 'isole', 'nord-ovest'],
    'germany': ['arnsberg-regbez', 'hessen', 'rheinland-pfalz', 'karlsruhe-regbez', 'saarland', 'koeln-regbez', 'sachsen-anhalt',
                'berlin', 'mecklenburg-vorpommern', 'sachsen', 'brandenburg', 'mittelfranken', 'schleswig-holstein', 'bremen', 'muenster-regbez',
                'schwaben', 'detmold-regbez', 'niederbayern', 'stuttgart-regbez', 'niedersachsen', 'thueringen', 'tuebingen-regbez', 'duesseldorf-regbez',
                'oberbayern', 'unterfranken', 'freiburg-regbez', 'oberfranken', 'hamburg', 'oberpfalz'],
    'netherlands': ['drenthe', 'noord-brabant', 'flevoland', 'noord-holland', 'friesland', 'overijssel', 'gelderland', 'utrecht', 'groningen', 
                'zeeland', 'limburg', 'zuid-holland'],
    'russia': ['central-fed-district', 'northwestern-fed-district', 'crimean-fed-district', 'siberian-fed-district', 'far-eastern-fed-district', 
            'south-fed-district', 'kaliningrad', 'north-caucasus-fed-district', 'volga-fed-district', 'ural-fed-district'],
    'spain': ['andalucia', 'galicia', 'aragon', 'islas-baleares', 'asturias', 'la-rioja', 'cantabria', 'madrid', 'castilla-la-mancha', 'melilla', 
            'castilla-y-leon', 'murcia', 'cataluna', 'navarra', 'ceuta', 'pais-vasco', 'valencia', 'extremadura'],
    'united-kingdom': ['england', 'scotland', 'wales'],
    'france': ['alsace', 'corse', 'midi-pyrenees', 'aquitaine', 'franche-comte', 'nord-pas-de-calais', 'auvergne', 'guyane', 'pays-de-la-loire', 
            'basse-normandie', 'haute-normandie', 'picardie', 'bourgogne', 'ile-de-france', 'poitou-charentes', 'bretagne', 'languedoc-roussillon', 
            'provence-alpes-cote-d-azur', 'centre', 'limousin', 'rhone-alpes', 'champagne-ardenne', 'lorraine'],
    'poland': ['dolnoslaskie', 'mazowieckie', 'swietokrzyskie', 'kujawsko-pomorskie', 'opolskie', 'warminsko-mazurskie', 'lodzkie', 'podkarpackie', 
            'wielkopolskie', 'lubelskie', 'podlaskie', 'zachodniopomorskie', 'lubuskie', 'pomorskie', 'malopolskie', 'slaskie'],
    'asia': ["afghanistan", "anhui", "armenia", "azerbaijan", "bangladesh", "beijing", "bhutan", "cambodia", "central-fed-district", "central-zone", 
            "chongqing", "chubu", "chugoku", "crimean-fed-district", "eastern-zone", "east-timor", "far-eastern-fed-district", "fujian", "gansu", 
            "gcc-states", "guangdong", "guangxi", "guizhou", "hainan", "hebei", "heilongjiang", "henan", "hokkaido", "hong-kong", "hubei", "hunan", 
            "inner-mongolia", "iran", "iraq", "israel-and-palestine", "java", "jiangsu", "jiangxi", "jilin", "jordan", "kalimantan", "kaliningrad", 
            "kansai", "kanto", "kazakhstan", "kyrgyzstan", "kyushu", "laos", "lebanon", "liaoning", "macau", "malaysia-singapore-brunei", "maldives",
            "maluku", "mongolia", "myanmar", "nepal", "ningxia", "north-caucasus-fed-district", "north-eastern-zone", "northern-zone", "north-korea",
            "northwestern-fed-district", "nusa-tenggara", "pakistan", "papua", "philippines", "qinghai", "shaanxi", "shandong", "shanghai", "shanxi",
            "shikoku", "siberian-fed-district", "sichuan", "southern-zone", "south-fed-district", "south-korea", "sri-lanka", "sulawesi", "sumatra",
            "syria", "taiwan", "tajikistan", "thailand", "tianjin", "tibet", "tohoku", "turkmenistan", "ural-fed-district", "uzbekistan", "vietnam",
            "volga-fed-district", "western-zone", "xinjiang", "yemen", "yunnan", "zhejiang"],
    'australia-oceania': ['american-oceania', 'nauru', 'samo', 'australia', 'new-caledonia', 'solomon-islands', 'cook-islands', 'new-zealand', 'tokelau', 
            'fiji', 'niue', 'tonga', 'ile-de-clipperton', 'palau', 'tuvalu', 'kiribati', 'papua-new-guinea', 'vanuatu', 'marshall-islands', 
            'pitcairn-islands', 'wallis-et-futuna', 'micronesia', 'polynesie-francaise'],
    'central-america': ['bahamas', 'cuba', 'haiti-and-domrep', 'panama', 'belize', 'el-salvador', 'honduras', 'costa-rica', 'guatemala', 'jamaica'],
    'south-america': ['argentina', 'bolivia', 'chile', 'colombia', 'ecuador', 'guyana', 'paraguay', 'peru', 'suriname', 'uruguay', 'venezuela', 
        'sul', 'sudeste', 'norte', 'nordeste', 'centro-oeste'],
    'north-america': ['greenland', 'mexico'],
    'europe': ['albania', 'faroe-islands', 'malta', 'georgia', 'montenegro', 'andorra', 'austria', 'greece', 'norway', 'azores', 'guernsey-jersey', 'belarus', 'hungary', 'portugal',
            'belgium', 'iceland', 'romania', 'bosnia-herzegovina', 'ireland-and-northern-ireland', 'bulgaria', 'isle-of-man', 'serbia', 'croatia', 'slovakia',
            'cyprus', 'kosovo', 'slovenia', 'czech-republic', 'latvia', 'denmark', 'liechtenstein', 'sweden', 'lithuania', 'switzerland', 'luxembourg', 'turkey',
            'estonia', 'macedonia', 'ukraine', 'finland', 'moldova', 'monaco']
}