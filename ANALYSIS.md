# Windowsill Database Analysis

> Generated: 2026-02-16 20:48:56
> Database: `vector_distillery.duckdb` (13.02 GB)
> Key parameters: ZSCORE_THRESHOLD=2.0, REEF_MIN_DEPTH=2, ISLAND_MIN_DIMS_FOR_SUBDIVISION=2

## Pipeline Metrics

| Metric | Value |
|--------|-------|
| Total memberships | 2,568,135 |
| Avg members/dim | 2,819.0 |
| Avg dims/word | 17.5 |
| Word-reef affinity rows | 2,442,489 |
| Reef coverage (any depth) | 100.0% (146,697 / 146,698) |
| Reef coverage (depth >= 2) | 64.9% (95,213 / 146,698) |
| Reef IDF range | [1.09, 5.24] |
| Database file size | 13.02 GB |

## Hierarchy

- **5** archipelagos (gen-0)
- **67** islands (gen-1)
- **283** reefs (gen-2)
- Noise dims: gen-0=0, gen-1=0, gen-2=0

### Gen-0 Archipelagos

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 0 | scientific and technical nomenclature | 266 | 80,879 | -0.029 | -0.104 | 0.832 | 0.058 | 0.094 | 0.016 |
| 1 | human activities and infrastructure | 198 | 78,371 | -0.057 | -0.232 | 0.782 | 0.102 | 0.095 | 0.021 |
| 2 | negative states and degradation | 164 | 78,700 | 0.194 | -0.223 | 0.674 | 0.117 | 0.168 | 0.041 |
| 3 | qualities and abstractions | 147 | 76,997 | -0.091 | -0.23 | 0.693 | 0.098 | 0.148 | 0.062 |
| 4 | specialized activities and practices | 136 | 2,756 | None | None | 0.369 | 0.394 | 0.207 | 0.03 |

### Gen-1 Islands

#### scientific and technical nomenclature (archipelago 0)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 0 | medicine and chemistry | 29 | 34,378 | -0.016 | -0.169 | 0.825 | 0.068 | 0.094 | 0.013 |
| 1 | biological sciences taxonomy | 29 | 28,685 | -0.008 | 0.005 | 0.863 | 0.042 | 0.084 | 0.012 |
| 2 | natural history specimens | 28 | 27,670 | -0.051 | -0.028 | 0.872 | 0.039 | 0.076 | 0.013 |
| 3 | perception and classification | 27 | 34,298 | -0.007 | -0.156 | 0.795 | 0.062 | 0.119 | 0.024 |
| 4 | specialized technical domains | 21 | 28,465 | -0.011 | -0.074 | 0.83 | 0.052 | 0.099 | 0.018 |
| 5 | scientific regional nomenclature | 19 | 23,771 | -0.036 | -0.056 | 0.86 | 0.044 | 0.08 | 0.016 |
| 6 | anatomical and historical | 18 | 22,105 | -0.057 | -0.107 | 0.814 | 0.09 | 0.079 | 0.017 |
| 7 | culture and life sciences | 17 | 20,435 | -0.068 | -0.142 | 0.841 | 0.057 | 0.088 | 0.015 |
| 8 | material and cultural descriptors | 17 | 21,231 | -0.081 | -0.12 | 0.824 | 0.059 | 0.102 | 0.015 |
| 9 | applied sciences interventions | 15 | 20,679 | -0.049 | -0.11 | 0.799 | 0.08 | 0.103 | 0.018 |
| 10 | geographic and spatial references | 13 | 18,539 | 0.026 | -0.204 | 0.836 | 0.076 | 0.077 | 0.011 |
| 11 | botanical and cultural classification | 12 | 17,756 | -0.003 | -0.119 | 0.826 | 0.051 | 0.105 | 0.018 |
| 12 | pathology and history | 9 | 12,713 | 0.041 | -0.121 | 0.821 | 0.044 | 0.112 | 0.023 |
| 13 | earth and human systems | 8 | 12,996 | -0.089 | -0.215 | 0.799 | 0.055 | 0.13 | 0.016 |
| 14 | descriptive taxonomic compounds | 4 | 6,380 | -0.062 | -0.082 | 0.816 | 0.056 | 0.113 | 0.016 |
#### human activities and infrastructure (archipelago 1)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 15 | people and society | 25 | 33,544 | -0.031 | -0.262 | 0.784 | 0.094 | 0.1 | 0.022 |
| 16 | physical systems structures | 20 | 24,861 | -0.053 | -0.245 | 0.782 | 0.107 | 0.091 | 0.02 |
| 17 | materials and documentation | 18 | 25,773 | -0.177 | -0.286 | 0.747 | 0.124 | 0.099 | 0.03 |
| 18 | sound and culture | 18 | 25,423 | -0.105 | -0.278 | 0.796 | 0.076 | 0.101 | 0.026 |
| 19 | domestic architecture spaces | 18 | 23,414 | 0.001 | -0.274 | 0.773 | 0.111 | 0.099 | 0.017 |
| 20 | deterioration and loss | 15 | 21,916 | -0.02 | -0.208 | 0.737 | 0.133 | 0.104 | 0.026 |
| 21 | industry and abstractions | 15 | 22,674 | -0.121 | -0.219 | 0.79 | 0.105 | 0.09 | 0.015 |
| 22 | occupational roles transportation | 14 | 17,455 | -0.041 | -0.153 | 0.828 | 0.093 | 0.068 | 0.011 |
| 23 | commerce and materials | 13 | 19,687 | -0.146 | -0.205 | 0.792 | 0.102 | 0.086 | 0.021 |
| 24 | conflict and quantities | 12 | 19,075 | -0.004 | -0.176 | 0.772 | 0.104 | 0.108 | 0.015 |
| 25 | power and conflict | 12 | 17,463 | 0.0 | -0.212 | 0.802 | 0.09 | 0.093 | 0.016 |
| 26 | historical cultural terms | 9 | 15,239 | -0.069 | -0.189 | 0.762 | 0.105 | 0.107 | 0.026 |
| 27 | tradition and expression | 9 | 11,711 | 0.107 | -0.203 | 0.821 | 0.078 | 0.081 | 0.02 |
#### negative states and degradation (archipelago 2)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 28 | structure and anomaly | 22 | 31,862 | 0.236 | -0.253 | 0.663 | 0.103 | 0.19 | 0.044 |
| 29 | emotion and hardship | 20 | 29,628 | 0.158 | -0.185 | 0.692 | 0.103 | 0.165 | 0.04 |
| 30 | contamination and transgression | 19 | 28,298 | 0.094 | -0.198 | 0.727 | 0.11 | 0.128 | 0.035 |
| 31 | dishonesty and impropriety | 18 | 27,220 | 0.306 | -0.173 | 0.648 | 0.079 | 0.202 | 0.071 |
| 33 | extremes and modification | 14 | 23,286 | 0.058 | -0.225 | 0.691 | 0.118 | 0.151 | 0.04 |
| 32 | absence and lack | 14 | 20,123 | 0.248 | -0.273 | 0.665 | 0.14 | 0.171 | 0.024 |
| 34 | bodily harm and crisis | 13 | 21,885 | 0.122 | -0.229 | 0.643 | 0.17 | 0.151 | 0.036 |
| 35 | linguistic modifiers and assessment | 13 | 22,084 | 0.18 | -0.229 | 0.63 | 0.149 | 0.185 | 0.036 |
| 36 | negative attributes and appearance | 10 | 17,131 | 0.241 | -0.237 | 0.665 | 0.13 | 0.168 | 0.036 |
| 37 | rejection and failure | 9 | 15,318 | 0.38 | -0.201 | 0.686 | 0.083 | 0.185 | 0.046 |
| 38 | symbols and decay | 8 | 13,635 | 0.042 | -0.329 | 0.69 | 0.143 | 0.133 | 0.033 |
| 39 | absence and biological processes | 4 | 6,703 | 0.506 | -0.153 | 0.7 | 0.08 | 0.197 | 0.023 |
#### qualities and abstractions (archipelago 3)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 40 | formal systems and conventions | 23 | 32,120 | -0.174 | -0.274 | 0.702 | 0.123 | 0.135 | 0.04 |
| 41 | capability and feasibility | 14 | 23,598 | -0.065 | -0.243 | 0.655 | 0.075 | 0.191 | 0.079 |
| 42 | scale and grandeur | 13 | 19,954 | -0.024 | -0.236 | 0.75 | 0.083 | 0.12 | 0.047 |
| 43 | nature and character | 12 | 20,591 | -0.115 | -0.162 | 0.636 | 0.082 | 0.161 | 0.121 |
| 44 | intensity and distinction | 12 | 19,768 | -0.142 | -0.194 | 0.702 | 0.095 | 0.15 | 0.053 |
| 45 | emotion and affect | 11 | 18,534 | -0.133 | -0.179 | 0.655 | 0.122 | 0.164 | 0.059 |
| 46 | structure and relationships | 11 | 16,034 | -0.054 | -0.236 | 0.675 | 0.13 | 0.146 | 0.049 |
| 47 | positive social interaction | 11 | 17,518 | -0.196 | -0.215 | 0.736 | 0.089 | 0.123 | 0.052 |
| 48 | negativity and harm | 10 | 17,456 | -0.062 | -0.171 | 0.711 | 0.084 | 0.145 | 0.06 |
| 49 | visibility and display | 9 | 15,613 | -0.007 | -0.226 | 0.626 | 0.146 | 0.176 | 0.051 |
| 50 | organizations and names | 7 | 10,632 | -0.036 | -0.38 | 0.785 | 0.06 | 0.119 | 0.036 |
| 51 | manner and direction | 6 | 11,131 | 0.097 | -0.221 | 0.723 | 0.055 | 0.147 | 0.075 |
| 52 | temperament and energy | 5 | 8,827 | -0.055 | -0.247 | 0.652 | 0.097 | 0.149 | 0.102 |
| 53 | manner and abruptness | 3 | 5,590 | -0.021 | -0.278 | 0.712 | 0.068 | 0.131 | 0.089 |
#### specialized activities and practices (archipelago 4)

| ID | Name | Dims | Words | Valence | Specificity | Noun | Verb | Adj | Adv |
|----|------|------|-------|---------|-------------|------|------|-----|-----|
| 54 | functional modifiers and states | 23 | 524 | None | None | 0.292 | 0.264 | 0.343 | 0.101 |
| 55 | sports and creative activities | 17 | 521 | None | None | 0.391 | 0.445 | 0.145 | 0.018 |
| 56 | performance and recording arts | 15 | 751 | None | None | 0.364 | 0.416 | 0.207 | 0.014 |
| 57 | constructed forms and performance | 14 | 203 | None | None | 0.391 | 0.459 | 0.146 | 0.004 |
| 58 | specialized domain terminology | 12 | 140 | None | None | 0.399 | 0.37 | 0.214 | 0.017 |
| 59 | games and systematic processes | 10 | 154 | None | None | 0.39 | 0.446 | 0.161 | 0.003 |
| 60 | physical techniques and crafts | 8 | 120 | None | None | 0.42 | 0.481 | 0.094 | 0.004 |
| 61 | bodily systems and signals | 8 | 84 | None | None | 0.404 | 0.363 | 0.216 | 0.018 |
| 62 | military and forceful action | 7 | 434 | None | None | 0.359 | 0.432 | 0.188 | 0.02 |
| 63 | sports and material crafts | 7 | 50 | None | None | 0.352 | 0.442 | 0.206 | 0.0 |
| 64 | transmission and bodily control | 7 | 81 | None | None | 0.382 | 0.394 | 0.212 | 0.012 |
| 65 | strategic games and vernacular | 6 | 139 | None | None | 0.427 | 0.365 | 0.191 | 0.017 |
| 66 | temporal measurement | 2 | 12 | None | None | 0.226 | 0.271 | 0.297 | 0.206 |

### Gen-2 Reefs

#### medicine and chemistry (island 0)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 2 | pharmaceutical compounds and sedatives | 6 | 10,104 | -0.081 | -0.166 |
| 0 | digestive and infectious disease | 5 | 8,233 | 0.128 | -0.138 |
| 1 | nuclear physics and radiometry | 4 | 6,123 | -0.23 | -0.239 |
| 5 | heat and chemical combustion | 4 | 7,018 | 0.148 | -0.134 |
| 3 | blood properties and theology | 3 | 4,268 | 0.157 | -0.137 |
| 6 | vascular surgery and catheterization | 3 | 4,599 | -0.126 | -0.214 |
| 4 | botanical classification and adoption | 2 | 3,435 | 0.105 | -0.17 |
| 7 | visual imagery and fabric | 2 | 3,392 | -0.299 | -0.158 |
#### biological sciences taxonomy (island 1)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 13 | botanical ecology biology | 6 | 8,520 | -0.093 | -0.042 |
| 14 | plant characteristics traits | 6 | 7,251 | 0.008 | 0.003 |
| 10 | flowering plants shrubs | 5 | 5,350 | -0.128 | 0.177 |
| 12 | scientific measurement terminology | 5 | 6,245 | 0.179 | 0.019 |
| 9 | anatomical body structures | 4 | 5,224 | 0.037 | -0.094 |
| 11 | philosophical abstract concepts | 3 | 4,605 | -0.042 | -0.072 |
#### natural history specimens (island 2)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 20 | insect and arthropod families | 6 | 7,659 | -0.066 | 0.038 |
| 15 | family and kinship | 5 | 6,114 | -0.139 | -0.108 |
| 17 | plant and hair textures | 4 | 4,526 | 0.059 | 0.035 |
| 21 | botanical and ancient studies | 4 | 5,007 | -0.14 | 0.092 |
| 16 | housing and bodily spaces | 3 | 4,391 | 0.03 | -0.254 |
| 18 | fossils and minerals | 3 | 4,140 | -0.129 | 0.005 |
| 19 | engineering and oddities | 3 | 3,979 | 0.093 | -0.076 |
#### perception and classification (island 3)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 22 | eye and vision terms | 6 | 9,933 | 0.006 | -0.211 |
| 23 | veneration and relics | 4 | 6,784 | -0.017 | -0.159 |
| 24 | harassment and hostility | 4 | 7,816 | 0.104 | -0.101 |
| 25 | physical appearance descriptors | 4 | 6,241 | -0.04 | -0.124 |
| 28 | scientific taxonomy | 4 | 7,170 | 0.061 | -0.132 |
| 26 | large-scale and widespread | 3 | 5,015 | -0.172 | -0.201 |
| 27 | mysticism and alchemy | 2 | 2,759 | -0.075 | -0.14 |
#### specialized technical domains (island 4)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 29 | taxonomic genera terms | 3 | 4,765 | -0.178 | -0.001 |
| 30 | educational institutions systems | 3 | 5,029 | -0.094 | -0.137 |
| 31 | curved spatial forms | 3 | 6,278 | 0.199 | -0.054 |
| 32 | visual imagery symbolism | 3 | 5,490 | -0.037 | -0.163 |
| 33 | tropical flora fauna | 3 | 4,368 | 0.194 | 0.091 |
| 34 | diplomatic assertion manner | 3 | 6,136 | -0.109 | -0.123 |
| 35 | medical technical materials | 3 | 5,369 | -0.052 | -0.129 |
#### scientific regional nomenclature (island 5)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 39 | chemical technical compounds | 4 | 7,080 | -0.023 | -0.086 |
| 38 | medical pharmaceutical terms | 3 | 4,500 | -0.048 | -0.109 |
| 40 | geographic taxonomic names | 3 | 3,809 | -0.1 | -0.06 |
| 41 | botanical culinary terms | 3 | 5,091 | -0.124 | -0.16 |
| 36 | asian regional places | 2 | 2,380 | 0.112 | 0.035 |
| 37 | australian birds flora | 2 | 2,038 | -0.059 | 0.144 |
| 42 | geological landscape features | 2 | 4,629 | 0.054 | -0.047 |
#### anatomical and historical (island 6)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 46 | body parts and fibers | 5 | 8,531 | 0.006 | -0.165 |
| 45 | anatomical and botanical terms | 4 | 5,976 | -0.107 | -0.071 |
| 47 | classical and prehistoric references | 4 | 6,111 | -0.082 | -0.036 |
| 44 | words containing g | 3 | 4,248 | 0.011 | -0.153 |
| 43 | physical movement intersections | 2 | 1,241 | -0.28 | -0.105 |
#### culture and life sciences (island 7)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 48 | food and dining | 5 | 6,565 | -0.098 | -0.165 |
| 49 | bacteria and anatomy | 4 | 6,470 | -0.053 | 0.0 |
| 50 | tools and decoration | 3 | 4,129 | -0.035 | -0.176 |
| 51 | language and writing | 3 | 4,154 | -0.032 | -0.204 |
| 52 | academia and theory | 2 | 2,665 | -0.125 | -0.224 |
#### material and cultural descriptors (island 8)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 53 | archaic geographic terms | 4 | 6,238 | -0.031 | -0.144 |
| 54 | color compound descriptors | 4 | 5,477 | -0.214 | -0.188 |
| 55 | cultural institutions and places | 4 | 5,993 | -0.066 | -0.095 |
| 56 | ornamental physical features | 3 | 3,905 | -0.052 | -0.011 |
| 57 | mineral and metallurgical terms | 2 | 3,454 | 0.013 | -0.149 |
#### applied sciences interventions (island 9)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 61 | geographic historical terms | 5 | 8,062 | 0.106 | -0.095 |
| 60 | chemical compounds elements | 4 | 6,269 | -0.196 | -0.11 |
| 58 | physical formation structure | 3 | 4,779 | -0.196 | -0.091 |
| 59 | medical procedures treatments | 3 | 4,494 | 0.033 | -0.156 |
#### geographic and spatial references (island 10)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 62 | uncommon proper nouns | 3 | 6,005 | 0.121 | -0.191 |
| 63 | measurements and locations | 3 | 4,727 | -0.067 | -0.162 |
| 65 | cities and urban geography | 3 | 5,088 | 0.078 | -0.18 |
| 64 | compound four words | 2 | 2,572 | -0.005 | -0.286 |
| 66 | nautical and maritime terms | 2 | 3,289 | -0.028 | -0.238 |
#### botanical and cultural classification (island 11)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 68 | botanical taxonomy | 4 | 8,071 | 0.008 | -0.045 |
| 67 | abstract qualities | 3 | 4,366 | 0.128 | -0.146 |
| 70 | cultural groups and time | 3 | 4,960 | -0.072 | -0.175 |
| 69 | food and place names | 2 | 2,931 | -0.118 | -0.144 |
#### pathology and history (island 12)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 71 | medical conditions and symptoms | 3 | 4,817 | 0.211 | -0.005 |
| 72 | neurological diseases and tumors | 3 | 4,771 | -0.087 | -0.077 |
| 73 | historical periods and events | 3 | 4,286 | -0.002 | -0.281 |
#### earth and human systems (island 13)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 74 | geological sediment indexing | 2 | 3,604 | 0.102 | -0.118 |
| 75 | anatomical body parts | 2 | 3,803 | -0.025 | -0.201 |
| 76 | climate and atmosphere | 2 | 3,231 | -0.222 | -0.334 |
| 77 | historical human civilization | 2 | 3,947 | -0.21 | -0.21 |
#### descriptive taxonomic compounds (island 14)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 78 | hyphenated color descriptors | 2 | 3,076 | -0.013 | -0.177 |
| 79 | taxonomic and cultural terms | 2 | 3,449 | -0.111 | 0.013 |
#### people and society (island 15)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 80 | travel and demonstration | 4 | 8,254 | -0.045 | -0.248 |
| 85 | agents and observers | 4 | 8,731 | 0.052 | -0.29 |
| 87 | workers and occupations | 4 | 6,653 | 0.042 | -0.325 |
| 82 | casual slang terms | 3 | 4,593 | -0.146 | -0.16 |
| 83 | historical figures places | 3 | 6,088 | -0.073 | -0.253 |
| 84 | competitive force framework | 3 | 5,506 | 0.111 | -0.296 |
| 81 | medical anatomy prosthetics | 2 | 3,352 | -0.199 | -0.314 |
| 86 | birth and origins | 2 | 3,845 | -0.122 | -0.17 |
#### physical systems structures (island 16)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 88 | hierarchy and classification | 3 | 4,244 | -0.127 | -0.188 |
| 89 | movement and transportation | 3 | 4,675 | -0.158 | -0.274 |
| 90 | trees and wooden structures | 3 | 4,056 | 0.034 | -0.227 |
| 91 | tools and implements | 3 | 4,867 | -0.079 | -0.308 |
| 92 | mental states and traits | 3 | 5,240 | 0.076 | -0.26 |
| 93 | waves and oscillation | 3 | 5,211 | 0.061 | -0.214 |
| 94 | attention and direction | 2 | 3,528 | -0.243 | -0.245 |
#### materials and documentation (island 17)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 96 | positive anticipation and preparation | 4 | 7,381 | -0.379 | -0.311 |
| 100 | oceanic regions and fauna | 4 | 7,105 | -0.098 | -0.236 |
| 97 | written text and books | 3 | 4,612 | -0.09 | -0.383 |
| 98 | textiles and sacred texts | 3 | 6,183 | -0.18 | -0.3 |
| 95 | physical insertion and joining | 2 | 3,633 | -0.18 | -0.276 |
| 99 | shapes and orientations | 2 | 3,809 | -0.057 | -0.183 |
#### sound and culture (island 18)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 101 | anatomy and sound | 4 | 6,546 | 0.014 | -0.188 |
| 102 | places and names | 4 | 6,523 | 0.055 | -0.344 |
| 103 | musical instruments and tones | 4 | 7,170 | -0.164 | -0.32 |
| 104 | technical objects and measurements | 3 | 4,514 | -0.268 | -0.349 |
| 105 | religious figures and titles | 3 | 6,763 | -0.238 | -0.185 |
#### domestic architecture spaces (island 19)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 106 | domestic spaces and furnishings | 7 | 11,587 | 0.088 | -0.239 |
| 110 | constraint and stillness | 4 | 6,562 | -0.139 | -0.248 |
| 107 | structure and assembly | 3 | 4,053 | -0.003 | -0.36 |
| 109 | insects and directional movement | 3 | 5,465 | -0.011 | -0.303 |
| 108 | yiddish colloquialisms | 1 | 7 | None | None |
#### deterioration and loss (island 20)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 114 | mumbling and ambling | 5 | 9,510 | -0.053 | -0.217 |
| 111 | emptiness and endings | 3 | 4,147 | 0.021 | -0.182 |
| 115 | forgetting and divesting | 3 | 5,823 | 0.041 | -0.27 |
| 112 | clicking and scraping | 2 | 3,097 | 0.054 | -0.172 |
| 113 | religious ideology themes | 2 | 3,556 | -0.166 | -0.166 |
#### industry and abstractions (island 21)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 116 | people and professions | 4 | 7,645 | -0.097 | -0.268 |
| 117 | technical compounds and elements | 3 | 4,863 | -0.23 | -0.151 |
| 118 | flowing and industrial processes | 3 | 5,380 | -0.204 | -0.176 |
| 120 | boundless and numerical concepts | 3 | 5,508 | -0.032 | -0.266 |
| 119 | death and lifelessness | 2 | 3,820 | -0.013 | -0.219 |
#### occupational roles transportation (island 22)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 123 | people and vehicles | 7 | 10,490 | -0.056 | -0.158 |
| 122 | compound occupational terms | 5 | 7,533 | -0.027 | -0.148 |
| 121 | informal nicknames slang | 2 | 1,236 | -0.001 | -0.142 |
#### commerce and materials (island 23)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 124 | technical units and measurements | 3 | 5,251 | -0.031 | -0.234 |
| 125 | mineral and material properties | 3 | 5,961 | -0.091 | -0.161 |
| 128 | financial and business transactions | 3 | 6,198 | -0.033 | -0.343 |
| 126 | seasonal and geographical terms | 2 | 2,765 | -0.39 | -0.112 |
| 127 | competitive awards and monuments | 2 | 2,729 | -0.325 | -0.114 |
#### conflict and quantities (island 24)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 129 | weapons hunting conflict | 7 | 11,039 | 0.006 | -0.172 |
| 130 | measurements raised quantities | 3 | 5,495 | 0.029 | -0.192 |
| 131 | materials and broadness | 2 | 4,694 | -0.087 | -0.162 |
#### power and conflict (island 25)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 135 | occupational roles operators | 4 | 7,044 | 0.016 | -0.331 |
| 132 | military assault defense | 3 | 4,453 | -0.028 | -0.21 |
| 133 | victims and oppressed | 3 | 6,046 | -0.019 | -0.086 |
| 134 | coastal geography numbers | 2 | 2,717 | 0.042 | -0.165 |
#### historical cultural terms (island 26)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 136 | archaic objects and titles | 3 | 5,152 | -0.048 | -0.199 |
| 137 | medical procedures and equipment | 2 | 3,599 | -0.049 | -0.21 |
| 138 | yiddish and german terms | 2 | 4,551 | -0.107 | -0.174 |
| 139 | rustic establishments and travel | 2 | 4,022 | -0.084 | -0.169 |
#### tradition and expression (island 27)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 140 | clothing and horses | 3 | 3,752 | 0.074 | -0.112 |
| 141 | legal and administrative | 2 | 2,463 | 0.243 | -0.259 |
| 142 | outdated and modernization | 2 | 3,094 | 0.121 | -0.317 |
| 143 | exclamations and whimsy | 2 | 3,699 | 0.008 | -0.167 |
#### structure and anomaly (island 28)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 144 | spatial arrangement and deformation | 6 | 11,531 | 0.172 | -0.209 |
| 146 | neutrality and nonaffiliation | 5 | 9,401 | 0.278 | -0.297 |
| 148 | unnaturalness and irrationality | 5 | 10,018 | 0.342 | -0.284 |
| 145 | biological systems and structure | 3 | 4,857 | 0.235 | -0.219 |
| 147 | physical fragility and embedding | 3 | 6,270 | 0.116 | -0.248 |
#### emotion and hardship (island 29)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 150 | discontent and doubt | 7 | 14,254 | 0.112 | -0.193 |
| 149 | cruelty and madness | 4 | 6,836 | 0.299 | -0.243 |
| 151 | geological features weather | 3 | 4,023 | 0.145 | -0.084 |
| 152 | intensity and difficulty | 3 | 5,910 | 0.114 | -0.163 |
| 153 | forgiveness and honor | 3 | 6,188 | 0.134 | -0.209 |
#### contamination and transgression (island 30)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 155 | reduction and degradation | 5 | 9,228 | 0.091 | -0.196 |
| 159 | cleaning and hygiene | 4 | 7,326 | 0.054 | -0.22 |
| 156 | danger and threat | 3 | 6,081 | 0.218 | -0.125 |
| 157 | crime and punishment | 3 | 5,804 | -0.018 | -0.28 |
| 158 | smell and sensory states | 3 | 5,453 | 0.104 | -0.151 |
| 154 | speech and human traits | 1 | 2,413 | 0.199 | -0.228 |
#### dishonesty and impropriety (island 31)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 164 | careless impropriety | 8 | 16,325 | 0.367 | -0.167 |
| 161 | deception and myth | 4 | 7,157 | 0.199 | -0.153 |
| 163 | biological processes growth | 3 | 5,436 | 0.42 | -0.185 |
| 162 | fraud and illegitimacy | 2 | 3,459 | 0.103 | -0.22 |
| 160 | classical traditional forms | 1 | 8 | None | None |
#### absence and lack (island 32)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 166 | unconstrained and unmeasured | 5 | 9,512 | 0.32 | -0.304 |
| 165 | thirst and hydration | 3 | 5,358 | 0.201 | -0.199 |
| 167 | co-prefix and emptiness | 3 | 3,554 | 0.151 | -0.349 |
| 168 | odorless and sterile | 3 | 5,140 | 0.241 | -0.246 |
#### extremes and modification (island 33)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 172 | excess and harm | 5 | 10,143 | 0.06 | -0.194 |
| 169 | perfection and restoration | 4 | 7,708 | 0.078 | -0.32 |
| 170 | physical dimensions | 3 | 5,668 | -0.03 | -0.13 |
| 171 | technology and combat | 2 | 3,820 | 0.145 | -0.254 |
#### bodily harm and crisis (island 34)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 176 | harm injury suffering | 4 | 8,240 | 0.083 | -0.331 |
| 173 | physical transformation change | 3 | 6,282 | 0.063 | -0.169 |
| 174 | sickness and affliction | 2 | 3,754 | 0.242 | -0.109 |
| 175 | crisis and distress | 2 | 3,955 | 0.219 | -0.241 |
| 177 | violent outburst expression | 2 | 4,363 | 0.073 | -0.225 |
#### linguistic modifiers and assessment (island 35)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 180 | negation and reversal | 5 | 10,375 | 0.199 | -0.221 |
| 178 | compound temporal modifiers | 3 | 5,230 | 0.177 | -0.225 |
| 179 | material assessment terms | 3 | 6,829 | 0.135 | -0.272 |
| 181 | informal colloquial nouns | 2 | 3,712 | 0.201 | -0.192 |
#### negative attributes and appearance (island 36)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 183 | negative qualities undesirable | 4 | 8,460 | 0.343 | -0.314 |
| 182 | conflict and hostility | 2 | 3,432 | 0.143 | -0.223 |
| 184 | purity and pretense | 2 | 3,979 | 0.179 | -0.19 |
| 185 | visual appearance color | 2 | 3,707 | 0.198 | -0.142 |
#### rejection and failure (island 37)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 186 | disapproval and dissuasion | 5 | 9,732 | 0.46 | -0.215 |
| 187 | deviance and rebellion | 2 | 3,603 | 0.402 | -0.187 |
| 188 | deficiency and deterioration | 2 | 3,862 | 0.159 | -0.182 |
#### symbols and decay (island 38)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 189 | written symbols and notation | 6 | 11,091 | 0.023 | -0.372 |
| 190 | decay and deterioration | 2 | 3,248 | 0.099 | -0.2 |
#### absence and biological processes (island 39)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 191 | negation and deprivation | 2 | 4,065 | 0.72 | -0.133 |
| 192 | biological processes and migration | 2 | 2,936 | 0.292 | -0.173 |
#### formal systems and conventions (island 40)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 195 | uncertainty and speculation | 4 | 7,067 | -0.227 | -0.233 |
| 199 | credentials and registration | 4 | 7,009 | -0.29 | -0.392 |
| 193 | classification and arrangement | 3 | 5,610 | 0.028 | -0.245 |
| 194 | religious and ceremonial | 3 | 5,488 | -0.25 | -0.241 |
| 196 | linguistic and phonetic | 3 | 6,569 | -0.159 | -0.28 |
| 197 | official positions and succession | 3 | 5,926 | -0.046 | -0.232 |
| 198 | emphasis and exacerbation | 3 | 6,585 | -0.223 | -0.264 |
#### capability and feasibility (island 41)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 202 | polymorphic and representational | 4 | 8,519 | -0.078 | -0.286 |
| 203 | dependable and effective | 4 | 9,247 | -0.264 | -0.222 |
| 200 | impossible and immeasurable | 3 | 5,659 | 0.073 | -0.123 |
| 201 | unknowable and unenforceable | 3 | 5,602 | 0.08 | -0.337 |
#### scale and grandeur (island 42)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 206 | extraordinary grandeur | 5 | 8,583 | 0.078 | -0.24 |
| 204 | measurement and precision | 3 | 5,396 | -0.04 | -0.296 |
| 205 | magnificence and splendor | 3 | 5,065 | -0.194 | -0.182 |
| 207 | questionable variability | 2 | 3,986 | 0.002 | -0.215 |
#### nature and character (island 43)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 210 | behavioral traits and demeanor | 4 | 8,380 | -0.167 | -0.171 |
| 208 | plants and botanical terms | 3 | 6,212 | -0.09 | -0.154 |
| 209 | manner and mental qualities | 3 | 6,604 | -0.041 | -0.223 |
| 211 | physical objects and places | 2 | 3,279 | -0.163 | -0.065 |
#### intensity and distinction (island 44)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 212 | acceleration and intensity | 3 | 5,614 | 0.045 | -0.219 |
| 213 | energetic movement and liveliness | 3 | 4,998 | -0.265 | -0.176 |
| 214 | texture and regional geography | 3 | 6,392 | -0.252 | -0.229 |
| 215 | excellence and renown | 3 | 5,886 | -0.095 | -0.151 |
#### emotion and affect (island 45)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 216 | desire and instinct | 4 | 8,744 | -0.062 | -0.181 |
| 217 | charm and delight | 3 | 5,402 | -0.375 | -0.201 |
| 218 | playful activity | 2 | 3,468 | -0.019 | -0.202 |
| 219 | fear and disgust | 2 | 3,884 | -0.027 | -0.12 |
#### structure and relationships (island 46)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 220 | qualities and characteristics | 3 | 4,903 | -0.04 | -0.199 |
| 221 | geometric and symmetrical | 3 | 3,614 | -0.035 | -0.246 |
| 222 | coexistence and cooperation | 3 | 6,068 | -0.072 | -0.27 |
| 223 | formal institutions and contests | 2 | 3,605 | -0.067 | -0.232 |
#### positive social interaction (island 47)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 224 | desire and pursuit | 3 | 5,781 | -0.34 | -0.248 |
| 226 | pleasantness and emotion | 3 | 5,676 | 0.07 | -0.125 |
| 227 | encouragement and aid | 3 | 5,029 | -0.191 | -0.231 |
| 225 | normalcy and physicality | 2 | 3,863 | -0.386 | -0.275 |
#### negativity and harm (island 48)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 228 | deterioration and harm | 3 | 5,808 | -0.034 | -0.214 |
| 230 | tyranny and aggression | 3 | 5,970 | 0.035 | -0.145 |
| 229 | obscure technical terms | 2 | 4,087 | -0.236 | -0.172 |
| 231 | personality and temperament | 2 | 3,953 | -0.074 | -0.146 |
#### visibility and display (island 49)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 233 | conspicuous attention-seeking | 4 | 8,880 | 0.135 | -0.246 |
| 232 | perception and demonstration | 3 | 6,404 | -0.057 | -0.214 |
| 234 | church attendance branches | 2 | 2,384 | -0.423 | -0.185 |
#### organizations and names (island 50)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 235 | institutions and administration | 5 | 8,160 | 0.035 | -0.409 |
| 236 | proper nouns and derivatives | 2 | 3,053 | -0.216 | -0.308 |
#### manner and direction (island 51)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 237 | adverbial qualities and manners | 4 | 8,062 | 0.101 | -0.199 |
| 238 | upward movement and elevation | 2 | 3,724 | 0.09 | -0.264 |
#### temperament and energy (island 52)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 239 | personality traits and dispositions | 3 | 5,777 | -0.046 | -0.269 |
| 240 | heat light and combustion | 2 | 3,514 | -0.067 | -0.213 |
#### manner and abruptness (island 53)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 241 | manner and abruptness | 3 | 5,590 | -0.021 | -0.278 |
#### functional modifiers and states (island 54)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 245 | comparative quality modifiers | 5 | 72 | None | None |
| 247 | grammatical and logical | 5 | 250 | None | None |
| 242 | positional and relational | 4 | 40 | None | None |
| 243 | normal usage patterns | 3 | 40 | None | None |
| 244 | state transformation physical | 3 | 106 | None | None |
| 246 | paired and doubled | 3 | 58 | None | None |
#### sports and creative activities (island 55)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 248 | inventory and supply | 4 | 154 | None | None |
| 249 | football plays | 3 | 49 | None | None |
| 250 | baseball and swimming | 3 | 219 | None | None |
| 251 | religious redemption kidnapping | 3 | 97 | None | None |
| 252 | computing and dance | 2 | 11 | None | None |
| 253 | writing and authorship | 2 | 29 | None | None |
#### performance and recording arts (island 56)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 255 | musical performance roles | 5 | 585 | None | None |
| 256 | visual recording media | 4 | 68 | None | None |
| 254 | medical treatment application | 3 | 95 | None | None |
| 257 | horse movement gaits | 3 | 30 | None | None |
#### constructed forms and performance (island 57)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 258 | architectural structures built | 5 | 75 | None | None |
| 260 | aircraft flight maneuvering | 4 | 64 | None | None |
| 259 | poetic verse rhythm | 3 | 41 | None | None |
| 261 | theatrical performance roles | 2 | 33 | None | None |
#### specialized domain terminology (island 58)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 262 | sailing and rhythm | 4 | 57 | None | None |
| 263 | animal domestication butchery | 3 | 56 | None | None |
| 264 | legal financial terms | 3 | 14 | None | None |
| 265 | autonomous political action | 2 | 18 | None | None |
#### games and systematic processes (island 59)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 268 | card games and bidding | 5 | 80 | None | None |
| 267 | statistics and sampling | 3 | 22 | None | None |
| 266 | agriculture and labor | 2 | 59 | None | None |
#### physical techniques and crafts (island 60)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 269 | sports techniques and moves | 4 | 61 | None | None |
| 270 | tap dancing footwork | 2 | 13 | None | None |
| 271 | weaving and inlay crafts | 2 | 52 | None | None |
#### bodily systems and signals (island 61)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 272 | sound and signal | 3 | 30 | None | None |
| 273 | anatomical structures joints | 3 | 44 | None | None |
| 274 | cricket terminology | 2 | 12 | None | None |
#### military and forceful action (island 62)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 275 | compound military terms | 4 | 422 | None | None |
| 276 | breaking through barriers | 3 | 18 | None | None |
#### sports and material crafts (island 63)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 278 | tennis and geology | 4 | 35 | None | None |
| 277 | tailoring and withering | 3 | 16 | None | None |
#### transmission and bodily control (island 64)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 279 | water air transmission | 4 | 57 | None | None |
| 280 | anatomical muscle control | 3 | 25 | None | None |
#### strategic games and vernacular (island 65)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 281 | chess moves and pieces | 4 | 61 | None | None |
| 282 | crude slang and vulgarity | 2 | 87 | None | None |
#### temporal measurement (island 66)

| ID | Name | Dims | Words | Valence | Spec |
|----|------|------|-------|---------|------|
| 283 | duration and length | 2 | 12 | None | None |

## Valence Analysis

| Metric | Value |
|--------|-------|
| Positive-pole dims (valence <= -0.15) | 184 |
| Negative-pole dims (valence >= 0.15) | 182 |
| Reef valence range | [-0.42, 0.72] |
| Negation vector pairs | 1639 |

**Top 5 positive-pole reefs** (most negative valence = negation decreases activation):

| Reef ID | Name | Valence |
|---------|------|---------|
| 234 | church attendance branches | -0.423 |
| 126 | seasonal and geographical terms | -0.39 |
| 225 | normalcy and physicality | -0.386 |
| 96 | positive anticipation and preparation | -0.379 |
| 217 | charm and delight | -0.375 |

**Top 5 negative-pole reefs** (most positive valence = negation increases activation):

| Reef ID | Name | Valence |
|---------|------|---------|
| 191 | negation and deprivation | 0.72 |
| 186 | disapproval and dissuasion | 0.46 |
| 163 | biological processes growth | 0.42 |
| 187 | deviance and rebellion | 0.402 |
| 164 | careless impropriety | 0.367 |

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
| Noun | 0.701 |
| Verb | 0.135 |
| Adj  | 0.133 |
| Adv  | 0.031 |

**Most verb-heavy reefs:**

| Reef ID | Name | Verb Frac | Dims |
|---------|------|-----------|------|
| 270 | tap dancing footwork | 0.5 | 2 |
| 271 | weaving and inlay crafts | 0.498 | 2 |
| 261 | theatrical performance roles | 0.478 | 2 |
| 248 | inventory and supply | 0.475 | 4 |
| 277 | tailoring and withering | 0.474 | 3 |

**Most adjective-heavy reefs:**

| Reef ID | Name | Adj Frac | Dims |
|---------|------|----------|------|
| 160 | classical traditional forms | 0.5 | 1 |
| 245 | comparative quality modifiers | 0.415 | 5 |
| 246 | paired and doubled | 0.371 | 3 |
| 243 | normal usage patterns | 0.34 | 3 |
| 247 | grammatical and logical | 0.335 | 5 |

## Universal Word Analytics

| Metric | Value |
|--------|-------|
| Universal words (specificity < 0) | 24,651 |
| Abstract dims (universal_pct >= 0.3) | 128 |
| Concrete dims (universal_pct <= 0.15) | 189 |
| Domain generals (arch_concentration >= 0.75) | 126 |
| Polysemy-inflated (sense_spread >= 15) | 326 |

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
| Exclusive word ratio (%) | 70.8 | 70.3 |
| Internal Jaccard | 0.0357 | 0.0287 |

## Reef Edges

| Metric | Value |
|--------|-------|
| Total reef edges | 58,998 |
| Containment range | [0.0, 1.0] |

## Word Variants

| Source | Count |
|--------|-------|
| base | 146,698 |
| morphy | 343,542 |
| **Total** | **490,240** |
