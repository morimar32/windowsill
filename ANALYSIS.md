# Windowsill Database Analysis

> Generated: 2026-02-16 13:11:53
> Database: `vector_distillery.duckdb` (12.95 GB)
> Key parameters: ZSCORE_THRESHOLD=2.0, REEF_MIN_DEPTH=2, ISLAND_MIN_DIMS_FOR_SUBDIVISION=2

## Pipeline Metrics

| Metric | Value |
|--------|-------|
| Total memberships | 2,564,399 |
| Avg members/dim | 3,339.1 |
| Avg dims/word | 17.5 |
| Word-reef affinity rows | 2,438,230 |
| Reef coverage (any depth) | 100.0% (146,697 / 146,698) |
| Reef coverage (depth >= 2) | 65.2% (95,637 / 146,698) |
| Reef IDF range | [0.94, 5.05] |
| Database file size | 12.95 GB |

## Hierarchy

- **4** archipelagos (gen-0)
- **52** islands (gen-1)
- **233** reefs (gen-2)
- Noise dims: gen-0=0, gen-1=0, gen-2=0

### Gen-0 Archipelagos

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 0 | scientific and descriptive terminology | 263 | 80,841 | -0.022 | -0.101 | 0.834 | 0.055 | 0.095 | 0.016 |
| 1 | human activities and material culture | 211 | 79,134 | -0.059 | -0.236 | 0.781 | 0.099 | 0.098 | 0.021 |
| 2 | negative states and abstract relations | 153 | 78,147 | 0.193 | -0.216 | 0.671 | 0.11 | 0.174 | 0.045 |
| 3 | qualities and evaluative attributes | 141 | 76,388 | -0.084 | -0.234 | 0.712 | 0.083 | 0.145 | 0.06 |

### Gen-1 Islands

#### scientific and descriptive terminology (archipelago 0)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 0 | life sciences and chemistry | 31 | 35,315 | 0.001 | -0.146 | 0.821 | 0.064 | 0.099 | 0.015 |
| 1 | biological classification systems | 26 | 25,660 | -0.028 | 0.021 | 0.876 | 0.038 | 0.074 | 0.012 |
| 2 | zoological and taxonomic classification | 26 | 26,181 | -0.101 | 0.011 | 0.875 | 0.04 | 0.072 | 0.013 |
| 3 | mysticism death and violence | 23 | 29,762 | 0.048 | -0.145 | 0.814 | 0.058 | 0.109 | 0.019 |
| 4 | historical natural nomenclature | 22 | 27,980 | -0.031 | -0.073 | 0.845 | 0.049 | 0.089 | 0.018 |
| 5 | visual perception and appearance | 20 | 24,663 | -0.061 | -0.134 | 0.814 | 0.057 | 0.108 | 0.021 |
| 6 | linguistic and etymological patterns | 19 | 26,007 | -0.009 | -0.079 | 0.835 | 0.051 | 0.098 | 0.016 |
| 7 | geography and built environments | 18 | 24,776 | -0.008 | -0.202 | 0.837 | 0.057 | 0.09 | 0.015 |
| 8 | specialized technical domains | 16 | 19,866 | -0.101 | -0.065 | 0.832 | 0.058 | 0.095 | 0.015 |
| 9 | anatomical and colloquial terminology | 16 | 22,033 | 0.03 | -0.118 | 0.811 | 0.069 | 0.104 | 0.016 |
| 10 | physical properties and metabolism | 12 | 16,697 | -0.072 | -0.105 | 0.823 | 0.061 | 0.096 | 0.02 |
| 11 | physical structures and medicine | 11 | 14,421 | 0.114 | -0.179 | 0.843 | 0.063 | 0.082 | 0.012 |
| 12 | physiological and technical systems | 10 | 15,778 | -0.018 | -0.117 | 0.788 | 0.062 | 0.13 | 0.02 |
| 13 | natural environments and history | 9 | 14,135 | -0.108 | -0.227 | 0.803 | 0.057 | 0.124 | 0.015 |
| 14 | physical objects and representations | 4 | 6,048 | 0.134 | -0.17 | 0.822 | 0.05 | 0.112 | 0.015 |
#### human activities and material culture (archipelago 1)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 15 | physical world navigation | 23 | 28,927 | -0.032 | -0.243 | 0.761 | 0.111 | 0.104 | 0.024 |
| 16 | perception and attributes | 21 | 28,435 | -0.228 | -0.267 | 0.738 | 0.134 | 0.102 | 0.026 |
| 17 | social transactions | 19 | 27,720 | -0.029 | -0.253 | 0.743 | 0.13 | 0.108 | 0.02 |
| 18 | material culture production | 18 | 26,117 | 0.0 | -0.233 | 0.758 | 0.107 | 0.108 | 0.027 |
| 19 | traditional labor domains | 17 | 21,669 | -0.039 | -0.181 | 0.834 | 0.079 | 0.075 | 0.013 |
| 20 | occupations and temperament | 17 | 24,155 | -0.076 | -0.242 | 0.801 | 0.085 | 0.093 | 0.021 |
| 21 | agency and adversity | 16 | 23,476 | 0.025 | -0.228 | 0.793 | 0.089 | 0.102 | 0.016 |
| 22 | technical and cultural terminology | 16 | 24,160 | -0.086 | -0.29 | 0.783 | 0.091 | 0.099 | 0.027 |
| 23 | domestic goods and processes | 14 | 20,821 | -0.022 | -0.259 | 0.764 | 0.106 | 0.107 | 0.023 |
| 24 | material objects and status | 13 | 18,540 | 0.024 | -0.172 | 0.798 | 0.086 | 0.1 | 0.016 |
| 25 | specialized trades and society | 11 | 17,093 | -0.092 | -0.209 | 0.823 | 0.066 | 0.095 | 0.016 |
| 26 | maritime and delicate elements | 10 | 16,040 | -0.138 | -0.217 | 0.779 | 0.096 | 0.098 | 0.027 |
| 27 | lineage and resources | 8 | 12,862 | -0.247 | -0.129 | 0.83 | 0.064 | 0.089 | 0.016 |
| 28 | temporal organization | 8 | 11,409 | 0.107 | -0.34 | 0.814 | 0.085 | 0.083 | 0.018 |
#### negative states and abstract relations (archipelago 2)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 29 | negative human characteristics | 23 | 32,749 | 0.173 | -0.187 | 0.669 | 0.091 | 0.176 | 0.064 |
| 30 | destruction and degradation | 19 | 27,852 | 0.108 | -0.19 | 0.725 | 0.116 | 0.129 | 0.03 |
| 31 | physical properties flow | 18 | 28,891 | 0.083 | -0.211 | 0.657 | 0.13 | 0.174 | 0.039 |
| 32 | boundaries and logic | 17 | 27,270 | 0.257 | -0.299 | 0.667 | 0.089 | 0.2 | 0.044 |
| 33 | cognitive and physical states | 15 | 24,992 | 0.288 | -0.229 | 0.617 | 0.148 | 0.203 | 0.032 |
| 34 | manner and modification | 14 | 23,130 | 0.1 | -0.215 | 0.689 | 0.112 | 0.15 | 0.049 |
| 35 | conflict and dysfunction | 11 | 18,268 | 0.202 | -0.295 | 0.698 | 0.13 | 0.135 | 0.037 |
| 36 | deception and social facade | 11 | 20,137 | 0.164 | -0.189 | 0.631 | 0.124 | 0.193 | 0.051 |
| 37 | morphological prefix patterns | 10 | 16,230 | 0.514 | -0.227 | 0.637 | 0.073 | 0.233 | 0.057 |
| 38 | flawed and deceptive | 8 | 13,794 | 0.192 | -0.148 | 0.685 | 0.076 | 0.169 | 0.07 |
| 39 | sensory and pathological | 7 | 11,093 | 0.179 | -0.156 | 0.704 | 0.113 | 0.158 | 0.025 |
#### qualities and evaluative attributes (archipelago 3)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 40 | human character traits | 23 | 32,231 | -0.193 | -0.186 | 0.705 | 0.096 | 0.143 | 0.056 |
| 41 | professional domains artifacts | 20 | 28,862 | -0.186 | -0.276 | 0.709 | 0.107 | 0.142 | 0.042 |
| 42 | properties and measurements | 15 | 25,227 | -0.042 | -0.238 | 0.667 | 0.075 | 0.192 | 0.066 |
| 43 | intentional conduct demeanor | 13 | 22,088 | -0.139 | -0.19 | 0.631 | 0.072 | 0.168 | 0.129 |
| 44 | scientific prestige authenticity | 13 | 20,129 | -0.057 | -0.237 | 0.734 | 0.085 | 0.131 | 0.05 |
| 45 | negative qualities deterioration | 12 | 20,006 | 0.036 | -0.185 | 0.716 | 0.071 | 0.153 | 0.06 |
| 46 | prediction expertise velocity | 11 | 17,048 | -0.144 | -0.264 | 0.76 | 0.088 | 0.11 | 0.041 |
| 47 | authority and commerce | 11 | 16,674 | 0.086 | -0.261 | 0.768 | 0.063 | 0.127 | 0.042 |
| 48 | excellence and extremity | 7 | 11,602 | 0.034 | -0.295 | 0.779 | 0.07 | 0.118 | 0.033 |
| 49 | structure mortality intensity | 6 | 9,907 | -0.035 | -0.217 | 0.682 | 0.075 | 0.169 | 0.074 |
| 50 | credibility and discernment | 6 | 9,857 | -0.06 | -0.299 | 0.767 | 0.066 | 0.121 | 0.047 |
| 51 | manner and disposition | 4 | 7,333 | -0.003 | -0.223 | 0.667 | 0.075 | 0.146 | 0.112 |

### Gen-2 Reefs

#### life sciences and chemistry (island 0)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 2 | biochemical properties reactions | 7 | 11,040 | 0.022 | -0.101 |
| 0 | medical conditions symptoms | 5 | 7,518 | 0.113 | -0.12 |
| 5 | pharmaceutical microbiology research | 5 | 7,369 | -0.112 | -0.09 |
| 6 | nuclear physics chemistry | 5 | 7,523 | -0.274 | -0.258 |
| 3 | infectious diseases disorders | 4 | 6,230 | 0.163 | -0.189 |
| 1 | chemical compounds substances | 3 | 5,637 | 0.13 | -0.132 |
| 4 | botanical seasonal adaptation | 2 | 3,435 | 0.105 | -0.17 |
#### biological classification systems (island 1)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 12 | flowering plants gardening | 7 | 7,265 | -0.165 | 0.104 |
| 8 | scientific taxonomy terminology | 6 | 8,330 | -0.069 | -0.004 |
| 11 | medical imaging techniques | 4 | 4,888 | 0.165 | 0.017 |
| 7 | fear and anxiety | 3 | 3,354 | -0.013 | -0.011 |
| 9 | physical coalescence processes | 3 | 4,319 | 0.036 | -0.015 |
| 10 | fungi and mycology | 3 | 3,942 | 0.038 | -0.049 |
#### zoological and taxonomic classification (island 2)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 15 | insect and arthropod families | 5 | 5,535 | -0.196 | 0.071 |
| 16 | botanical and marine classification | 5 | 6,470 | -0.185 | 0.061 |
| 13 | medical terms and taxonomy | 4 | 6,124 | 0.004 | -0.112 |
| 14 | birds fish and wildlife | 4 | 4,796 | -0.021 | -0.052 |
| 17 | classical antiquity and anatomy | 4 | 5,219 | -0.098 | 0.052 |
| 18 | animal husbandry and allergies | 4 | 5,221 | -0.064 | 0.021 |
#### mysticism death and violence (island 3)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 24 | parasites and anatomy | 6 | 9,003 | 0.022 | -0.121 |
| 22 | religion and territory | 5 | 8,547 | 0.076 | -0.206 |
| 23 | occult and chemistry | 5 | 7,928 | 0.02 | -0.149 |
| 21 | violence and blood | 4 | 7,148 | 0.157 | -0.124 |
| 19 | feelings and perception | 2 | 3,552 | 0.077 | -0.135 |
| 20 | flowers and burial | 1 | 1,404 | -0.277 | -0.073 |
#### historical natural nomenclature (island 4)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 25 | anatomical and historical terms | 5 | 7,887 | -0.037 | -0.125 |
| 26 | botanical and geographic names | 5 | 7,985 | -0.025 | -0.1 |
| 27 | archaic objects and tools | 4 | 8,145 | -0.047 | -0.042 |
| 28 | natural species and habitats | 3 | 2,795 | -0.054 | 0.117 |
| 30 | global places and flora | 3 | 5,321 | -0.007 | -0.104 |
| 29 | genus and taxa classifications | 2 | 3,204 | 0.004 | -0.172 |
#### visual perception and appearance (island 5)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 35 | body parts and appearance | 8 | 10,757 | -0.057 | -0.108 |
| 32 | color hues and shades | 5 | 7,530 | -0.146 | -0.141 |
| 34 | illumination and brightness | 3 | 4,663 | -0.127 | -0.105 |
| 31 | darkness and visibility | 2 | 3,194 | 0.085 | -0.199 |
| 33 | visual art and artists | 2 | 3,143 | 0.084 | -0.199 |
#### linguistic and etymological patterns (island 6)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 36 | latin prefixes de- car- | 4 | 7,289 | -0.045 | -0.127 |
| 37 | medical suffixes -osis -itis | 3 | 5,029 | -0.094 | -0.137 |
| 38 | technical abbreviations and compounds | 3 | 5,369 | -0.052 | -0.129 |
| 41 | classical and ancient terms | 3 | 5,300 | -0.125 | 0.028 |
| 39 | sc- words and insults | 2 | 4,470 | 0.193 | -0.072 |
| 40 | place names and botany | 2 | 2,806 | 0.111 | 0.104 |
| 42 | proper nouns geography | 2 | 3,014 | 0.106 | -0.169 |
#### geography and built environments (island 7)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 44 | spanish places and flora | 5 | 7,755 | -0.121 | -0.175 |
| 47 | cities and urban landscapes | 5 | 8,210 | 0.104 | -0.199 |
| 43 | classical architecture ideals | 4 | 7,239 | -0.023 | -0.165 |
| 45 | chemical and medical terms | 2 | 3,613 | -0.042 | -0.244 |
| 46 | northern european proper nouns | 2 | 3,542 | 0.06 | -0.307 |
#### specialized technical domains (island 8)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 48 | dinosaurs and neurons | 6 | 8,426 | -0.119 | -0.049 |
| 49 | geometric and musical | 6 | 8,168 | 0.007 | -0.065 |
| 50 | craft and construction | 4 | 5,937 | -0.234 | -0.089 |
#### anatomical and colloquial terminology (island 9)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 51 | taxonomic and technical terminology | 5 | 8,303 | -0.131 | -0.11 |
| 52 | medical conditions and clergy | 3 | 4,654 | -0.094 | -0.233 |
| 53 | informal speech and behavior | 3 | 5,681 | 0.113 | -0.061 |
| 54 | skeletal anatomy and appendages | 3 | 4,804 | 0.21 | -0.074 |
| 55 | hair and bodily oddities | 2 | 2,804 | 0.223 | -0.119 |
#### physical properties and metabolism (island 10)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 56 | dining and traditionalism | 5 | 7,040 | -0.009 | -0.126 |
| 58 | biological formation and metabolism | 4 | 6,377 | -0.119 | -0.063 |
| 57 | size and physical properties | 3 | 4,960 | -0.113 | -0.127 |
#### physical structures and medicine (island 11)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 59 | water and floating matter | 4 | 6,373 | 0.111 | -0.218 |
| 60 | body movement and anatomy | 3 | 4,302 | 0.097 | -0.255 |
| 61 | structures and medical equipment | 2 | 2,597 | 0.098 | -0.061 |
| 62 | medical procedures and devices | 2 | 2,647 | 0.159 | -0.103 |
#### physiological and technical systems (island 12)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 63 | typography and microbiology | 5 | 8,468 | -0.051 | -0.138 |
| 64 | visceral and metabolic conditions | 5 | 8,486 | 0.015 | -0.096 |
#### natural environments and history (island 13)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 65 | geological structures and forms | 3 | 5,003 | -0.02 | -0.184 |
| 66 | anatomical parts and vernacular | 2 | 3,803 | -0.025 | -0.201 |
| 67 | climate and atmospheric systems | 2 | 3,231 | -0.222 | -0.334 |
| 68 | historical and monumental themes | 2 | 3,947 | -0.21 | -0.21 |
#### physical objects and representations (island 14)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 69 | physical objects and depictions | 4 | 6,048 | 0.134 | -0.17 |
#### physical world navigation (island 15)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 70 | physical motion and oscillation | 5 | 8,721 | 0.029 | -0.249 |
| 75 | building exteriors and surfaces | 5 | 7,267 | -0.091 | -0.259 |
| 71 | branching structures and trees | 4 | 5,906 | -0.036 | -0.146 |
| 72 | travelers and practitioners | 3 | 6,350 | -0.025 | -0.246 |
| 73 | vehicles and riding | 3 | 4,428 | -0.087 | -0.234 |
| 74 | enclosed spaces and buoyancy | 3 | 4,612 | 0.02 | -0.343 |
#### perception and attributes (island 16)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 76 | personal gestures and appearance | 4 | 6,955 | -0.404 | -0.26 |
| 77 | formed structures and launching | 3 | 4,435 | -0.254 | -0.238 |
| 78 | speech and death | 3 | 4,667 | -0.31 | -0.262 |
| 79 | color and quantity | 3 | 5,508 | -0.032 | -0.266 |
| 80 | components and emphasis | 3 | 5,906 | -0.153 | -0.304 |
| 82 | enumeration and assignment | 3 | 5,920 | -0.214 | -0.313 |
| 81 | light and radiation | 2 | 3,465 | -0.139 | -0.21 |
#### social transactions (island 17)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 85 | cleansing and expulsion | 4 | 7,890 | 0.009 | -0.245 |
| 87 | commerce and exchange | 4 | 7,730 | 0.032 | -0.27 |
| 83 | silence and repose | 3 | 5,263 | 0.066 | -0.192 |
| 84 | places and structures | 3 | 5,131 | 0.004 | -0.334 |
| 86 | capture and decline | 3 | 5,130 | -0.216 | -0.257 |
| 88 | self and independence | 2 | 3,993 | -0.143 | -0.199 |
#### material culture production (island 18)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 90 | sound noise instruments | 4 | 6,757 | 0.115 | -0.187 |
| 89 | trees wood damage | 3 | 5,945 | 0.056 | -0.162 |
| 91 | cultural proper names | 3 | 4,818 | -0.11 | -0.254 |
| 92 | lights energy sports | 3 | 5,351 | 0.021 | -0.277 |
| 93 | fish assembly labor | 3 | 5,900 | 0.002 | -0.249 |
| 94 | writing text authorship | 2 | 4,294 | -0.18 | -0.308 |
#### traditional labor domains (island 19)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 95 | cutting and severing | 3 | 5,009 | -0.128 | -0.214 |
| 96 | athleticism and physical strength | 3 | 4,640 | 0.01 | -0.238 |
| 97 | maritime structures and rigging | 3 | 4,950 | -0.088 | -0.132 |
| 98 | livestock and animal husbandry | 3 | 4,619 | 0.044 | -0.126 |
| 99 | historical warfare and transport | 3 | 4,523 | 0.137 | -0.2 |
| 100 | aquatic creatures and wildlife | 2 | 2,841 | -0.294 | -0.171 |
#### occupations and temperament (island 20)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 106 | workers and tradespeople | 5 | 8,997 | -0.066 | -0.275 |
| 102 | rage and noxious fury | 3 | 5,956 | -0.03 | -0.224 |
| 105 | proper names and occupations | 3 | 4,958 | -0.009 | -0.178 |
| 101 | compression and alphabetization | 2 | 2,934 | -0.084 | -0.301 |
| 103 | cooking and food prep | 2 | 3,380 | -0.434 | -0.176 |
| 104 | social roles and behavior | 2 | 3,927 | 0.097 | -0.29 |
#### agency and adversity (island 21)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 109 | agents and performers | 5 | 9,591 | 0.037 | -0.307 |
| 107 | discomfort and oppression | 3 | 5,528 | 0.123 | -0.214 |
| 108 | intrusion and violation | 3 | 5,079 | 0.011 | -0.219 |
| 111 | professions and specialists | 3 | 5,757 | -0.084 | -0.272 |
| 110 | fragments and outlaws | 2 | 3,161 | 0.035 | 0.002 |
#### technical and cultural terminology (island 22)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 112 | proper nouns and locations | 3 | 5,820 | 0.045 | -0.311 |
| 113 | measurements and technical specs | 3 | 4,979 | -0.167 | -0.353 |
| 114 | german and directional terms | 3 | 5,146 | -0.004 | -0.338 |
| 115 | textiles and hindu terminology | 3 | 6,183 | -0.18 | -0.3 |
| 116 | astronomical and exotic terms | 2 | 4,237 | -0.114 | -0.157 |
| 117 | casual culture and recreation | 2 | 3,895 | -0.119 | -0.211 |
#### domestic goods and processes (island 23)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 118 | dwelling spaces furniture | 5 | 8,407 | 0.108 | -0.231 |
| 119 | manufactured goods trades | 3 | 5,082 | -0.059 | -0.349 |
| 120 | procedures and abstractions | 3 | 5,128 | -0.056 | -0.25 |
| 121 | rest and tranquility | 3 | 5,828 | -0.169 | -0.224 |
#### material objects and status (island 24)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 125 | clothing and equestrian | 5 | 7,287 | 0.132 | -0.211 |
| 123 | animals and body parts | 3 | 4,805 | 0.041 | -0.136 |
| 124 | nobility and social roles | 3 | 5,531 | -0.152 | -0.203 |
| 122 | texture and tactile | 2 | 3,697 | -0.009 | -0.085 |
#### specialized trades and society (island 25)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 126 | tools and craftsmen | 3 | 4,851 | -0.171 | -0.239 |
| 127 | social deviants and outcasts | 3 | 6,113 | -0.154 | -0.165 |
| 129 | accomplished professionals | 3 | 5,787 | -0.038 | -0.254 |
| 128 | coastal and frontier | 2 | 2,717 | 0.042 | -0.165 |
#### maritime and delicate elements (island 26)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 132 | maritime occupations places | 4 | 7,314 | -0.076 | -0.232 |
| 130 | fragile ethereal objects | 3 | 5,750 | -0.145 | -0.207 |
| 131 | movement and migration | 3 | 4,885 | -0.214 | -0.207 |
#### lineage and resources (island 27)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 133 | kinship and titles | 3 | 4,328 | -0.306 | -0.067 |
| 134 | proper names and heritage | 3 | 6,251 | -0.229 | -0.168 |
| 135 | industrial materials and agriculture | 2 | 3,447 | -0.186 | -0.162 |
#### temporal organization (island 28)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 136 | scheduling and temporal tracking | 5 | 7,035 | 0.216 | -0.369 |
| 137 | pace and progression | 3 | 5,086 | -0.074 | -0.291 |
#### negative human characteristics (island 29)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 139 | cruelty and victimization | 4 | 7,918 | 0.237 | -0.18 |
| 141 | disgusting dangerous qualities | 4 | 8,147 | 0.204 | -0.117 |
| 143 | passionate controversial youth | 4 | 8,581 | 0.068 | -0.206 |
| 138 | distinctive physical traits | 3 | 6,206 | 0.128 | -0.151 |
| 140 | epic narratives observers | 3 | 6,005 | 0.114 | -0.225 |
| 142 | deceit and heedlessness | 3 | 4,846 | 0.235 | -0.218 |
| 144 | technical scientific terms | 2 | 3,547 | 0.251 | -0.253 |
#### destruction and degradation (island 30)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 146 | hostility and harm | 5 | 9,983 | 0.106 | -0.169 |
| 145 | beverages and containers | 4 | 6,519 | 0.158 | -0.197 |
| 147 | military and containment | 3 | 5,901 | 0.048 | -0.24 |
| 149 | decay and dissolution | 3 | 6,246 | 0.176 | -0.173 |
| 148 | demolition and removal | 2 | 3,775 | 0.138 | -0.151 |
| 150 | contamination and discoloration | 2 | 3,519 | -0.025 | -0.214 |
#### physical properties flow (island 31)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 155 | dimensional proportions | 7 | 13,792 | 0.001 | -0.199 |
| 152 | physical accessibility traits | 5 | 9,808 | 0.095 | -0.213 |
| 151 | pulsation and propulsion | 3 | 6,028 | 0.185 | -0.18 |
| 154 | flow and conductivity | 3 | 5,999 | 0.156 | -0.265 |
#### boundaries and logic (island 32)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 160 | clarity and simplicity | 4 | 9,764 | 0.208 | -0.328 |
| 156 | prohibition and restriction | 3 | 6,307 | 0.21 | -0.421 |
| 157 | decomposition and structure | 3 | 5,781 | 0.271 | -0.215 |
| 159 | enclosure and enrichment | 3 | 5,902 | 0.258 | -0.226 |
| 158 | irrationality and critique | 2 | 3,992 | 0.42 | -0.319 |
| 161 | separation and division | 2 | 3,639 | 0.237 | -0.276 |
#### cognitive and physical states (island 33)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 162 | biology and learning | 4 | 7,577 | 0.193 | -0.189 |
| 163 | physical movement states | 3 | 5,132 | 0.389 | -0.118 |
| 165 | inverted and irregular | 3 | 7,336 | 0.326 | -0.243 |
| 166 | speculation and thought | 3 | 6,131 | 0.381 | -0.315 |
| 164 | sudden acquisition events | 2 | 4,689 | 0.128 | -0.327 |
#### manner and modification (island 34)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 168 | manner and frequency adverbs | 4 | 8,782 | 0.146 | -0.221 |
| 170 | modification and reconstruction | 4 | 7,997 | -0.106 | -0.145 |
| 167 | animal traits and deception | 3 | 5,208 | 0.152 | -0.179 |
| 169 | assertiveness and asymmetry | 3 | 5,440 | 0.26 | -0.336 |
#### conflict and dysfunction (island 35)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 172 | suffering and inadequacy | 4 | 8,303 | 0.217 | -0.383 |
| 171 | conflict and hostility | 3 | 5,182 | 0.147 | -0.235 |
| 173 | absurdity and nonsense | 2 | 3,594 | 0.404 | -0.235 |
| 174 | neural and biological systems | 2 | 4,159 | 0.053 | -0.269 |
#### deception and social facade (island 36)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 176 | social demeanor and affectation | 4 | 8,648 | 0.075 | -0.201 |
| 177 | dishonor and dispute | 4 | 9,146 | 0.259 | -0.212 |
| 175 | illusion and falseness | 3 | 5,550 | 0.156 | -0.144 |
#### morphological prefix patterns (island 37)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 178 | negative prefix modifiers | 7 | 12,229 | 0.644 | -0.195 |
| 180 | co- compound words | 2 | 3,553 | 0.151 | -0.349 |
| 179 | spec- technical terms | 1 | 2,039 | 0.325 | -0.203 |
#### flawed and deceptive (island 38)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 181 | deception and obscurity | 4 | 7,735 | 0.209 | -0.133 |
| 182 | clever peculiarity | 2 | 3,469 | 0.191 | -0.144 |
| 183 | defective and undesirable | 2 | 3,862 | 0.159 | -0.182 |
#### sensory and pathological (island 39)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 184 | sanitation and quietness | 3 | 5,185 | 0.201 | -0.212 |
| 185 | colors and temperatures | 2 | 3,508 | 0.026 | -0.136 |
| 186 | fungal disease conditions | 2 | 3,275 | 0.3 | -0.092 |
#### human character traits (island 40)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 187 | obsession and bodily instinct | 5 | 10,631 | -0.057 | -0.194 |
| 188 | encouragement and thrift | 3 | 5,029 | -0.191 | -0.231 |
| 189 | brash juvenility | 3 | 5,100 | -0.19 | -0.125 |
| 190 | marriage and respectability | 3 | 5,311 | -0.271 | -0.276 |
| 191 | emptiness and pleasure | 3 | 5,424 | -0.172 | -0.125 |
| 192 | entertaining engagement | 3 | 6,273 | -0.14 | -0.166 |
| 193 | eloquent excellence | 3 | 5,656 | -0.418 | -0.183 |
#### professional domains artifacts (island 41)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 194 | medical imaging radiology | 3 | 5,855 | -0.16 | -0.201 |
| 195 | physical objects tools | 3 | 6,200 | -0.223 | -0.316 |
| 199 | civic values clarity | 3 | 6,201 | -0.193 | -0.308 |
| 200 | literary artistic forms | 3 | 5,903 | -0.16 | -0.286 |
| 196 | manner and certainty | 2 | 4,082 | -0.219 | -0.202 |
| 197 | official roles status | 2 | 4,466 | -0.068 | -0.253 |
| 198 | claims and assertions | 2 | 3,350 | -0.235 | -0.264 |
| 201 | administrative documentation | 2 | 3,383 | -0.236 | -0.372 |
#### properties and measurements (island 42)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 204 | mutability and feasibility | 5 | 10,838 | -0.182 | -0.234 |
| 205 | physical properties and substance | 4 | 8,819 | -0.014 | -0.267 |
| 202 | standard norms and measures | 3 | 5,696 | 0.037 | -0.319 |
| 203 | extreme scale and intensity | 3 | 5,659 | 0.073 | -0.123 |
#### intentional conduct demeanor (island 43)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 208 | deliberate behavior and cunning | 6 | 11,869 | -0.062 | -0.234 |
| 206 | manner and physical qualities | 3 | 5,264 | -0.225 | -0.097 |
| 207 | acceptance and dedication | 2 | 4,791 | -0.441 | -0.235 |
| 209 | scandal and audacity | 2 | 4,628 | 0.063 | -0.153 |
#### scientific prestige authenticity (island 44)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 210 | chemical spectrum analysis | 3 | 4,577 | -0.103 | -0.206 |
| 211 | enormous size magnitude | 3 | 5,863 | 0.094 | -0.26 |
| 212 | fame glory achievement | 3 | 5,175 | 0.052 | -0.243 |
| 213 | occupational roles titles | 2 | 4,126 | -0.436 | -0.264 |
| 214 | dubious false deceptive | 2 | 3,986 | 0.002 | -0.215 |
#### negative qualities deterioration (island 45)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 215 | deterioration and decline | 3 | 5,660 | -0.011 | -0.212 |
| 216 | systemic tendencies flaws | 3 | 6,348 | 0.186 | -0.26 |
| 217 | malice and sensory | 3 | 6,068 | -0.038 | -0.163 |
| 218 | abstract intensity erudition | 3 | 5,389 | 0.005 | -0.107 |
#### prediction expertise velocity (island 46)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 219 | prediction and certainty | 4 | 6,371 | -0.062 | -0.283 |
| 220 | probability and theory | 3 | 5,764 | -0.186 | -0.255 |
| 221 | professional roles and membership | 2 | 3,643 | -0.053 | -0.247 |
| 222 | speed and virtue | 2 | 3,588 | -0.338 | -0.26 |
#### authority and commerce (island 47)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 223 | control and governance | 3 | 4,847 | 0.125 | -0.246 |
| 224 | manner and certainty adverbs | 3 | 5,656 | 0.257 | -0.2 |
| 225 | commerce and competition | 3 | 4,964 | -0.007 | -0.367 |
| 226 | theoretical compliance | 2 | 3,484 | -0.088 | -0.216 |
#### excellence and extremity (island 48)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 227 | notable achievement and excellence | 5 | 8,827 | 0.087 | -0.3 |
| 228 | maximum extent and modification | 2 | 3,290 | -0.098 | -0.283 |
#### structure mortality intensity (island 49)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 229 | symmetry and geometric structure | 2 | 3,612 | -0.035 | -0.246 |
| 230 | mortality and ephemeral existence | 2 | 2,807 | -0.074 | -0.12 |
| 231 | extreme degrees and intensity | 2 | 4,332 | 0.004 | -0.285 |
#### credibility and discernment (island 50)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 232 | credibility and discernment | 6 | 9,857 | -0.06 | -0.299 |
#### manner and disposition (island 51)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 233 | manner and disposition | 4 | 7,333 | -0.003 | -0.223 |

## Valence Analysis

| Metric | Value |
|--------|-------|
| Positive-pole dims (valence <= -0.15) | 184 |
| Negative-pole dims (valence >= 0.15) | 182 |
| Reef valence range | [-0.44, 0.64] |
| Negation vector pairs | 1639 |

**Top 5 positive-pole reefs** (most negative valence = negation decreases activation):

| Reef ID | Name | Valence |
|---------|------|---------|
| 207 | acceptance and dedication | -0.441 |
| 213 | occupational roles titles | -0.436 |
| 103 | cooking and food prep | -0.434 |
| 193 | eloquent excellence | -0.418 |
| 76 | personal gestures and appearance | -0.404 |

**Top 5 negative-pole reefs** (most positive valence = negation increases activation):

| Reef ID | Name | Valence |
|---------|------|---------|
| 178 | negative prefix modifiers | 0.644 |
| 158 | irrationality and critique | 0.42 |
| 173 | absurdity and nonsense | 0.404 |
| 163 | physical movement states | 0.389 |
| 166 | speculation and thought | 0.381 |

## Specificity Bands

| Band | Words |
|------|-------|
| -2 | 3,761 |
| -1 | 20,890 |
| 0 | 96,513 |
| 1 | 23,919 |
| 2 | 1,615 |

## POS Composition

**Corpus-level averages (across all dims):**

| POS | Avg Fraction |
|-----|-------------|
| Noun | 0.765 |
| Verb | 0.083 |
| Adj  | 0.121 |
| Adv  | 0.031 |

**Most verb-heavy reefs:**

| Reef ID | Name | Verb Frac | Dims |
|---------|------|-----------|------|
| 164 | sudden acquisition events | 0.252 | 2 |
| 80 | components and emphasis | 0.188 | 3 |
| 166 | speculation and thought | 0.177 | 3 |
| 85 | cleansing and expulsion | 0.16 | 4 |
| 154 | flow and conductivity | 0.153 | 3 |

**Most adjective-heavy reefs:**

| Reef ID | Name | Adj Frac | Dims |
|---------|------|----------|------|
| 160 | clarity and simplicity | 0.256 | 4 |
| 178 | negative prefix modifiers | 0.256 | 7 |
| 165 | inverted and irregular | 0.237 | 3 |
| 231 | extreme degrees and intensity | 0.226 | 2 |
| 177 | dishonor and dispute | 0.22 | 4 |

## Universal Word Analytics

| Metric | Value |
|--------|-------|
| Universal words (specificity < 0) | 24,651 |
| Abstract dims (universal_pct >= 0.3) | 128 |
| Concrete dims (universal_pct <= 0.15) | 46 |
| Domain generals (arch_concentration >= 0.75) | 111 |
| Polysemy-inflated (sense_spread >= 15) | 363 |

## Senses and Compounds

| Metric | Value |
|--------|-------|
| Total senses | 65,845 |
| Domain-anchored senses | 4,409 |
| Total compounds | 63,912 |
| Compositional | 28,784 |

## Reef Quality Summary

| Metric | Mean | Median |
|--------|------|--------|
| Exclusive word ratio (%) | 66.7 | 64.7 |
| Internal Jaccard | 0.0294 | 0.0276 |

## Reef Edges

| Metric | Value |
|--------|-------|
| Total reef edges | 37,202 |
| Containment range | [0.0, 0.267] |

## Word Variants

| Source | Count |
|--------|-------|
| base | 146,698 |
| morphy | 343,542 |
| **Total** | **490,240** |
