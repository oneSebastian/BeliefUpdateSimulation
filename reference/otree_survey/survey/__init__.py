from otree.api import *
import pandas as pd
from itertools import product
import itertools
import random

country_choices = [
        ('AF', 'Afghanistan'),
        ('AL', 'Albania'),
        ('DZ', 'Algeria'),
        ('AS', 'American Samoa'),
        ('AD', 'Andorra'),
        ('AO', 'Angola'),
        ('AI', 'Anguilla'),
        ('AQ', 'Antarctica'),
        ('AG', 'Antigua and Barbuda'),
        ('AR', 'Argentina'),
        ('AM', 'Armenia'),
        ('AW', 'Aruba'),
        ('AU', 'Australia'),
        ('AT', 'Austria'),
        ('AZ', 'Azerbaijan'),
        ('BS', 'Bahamas'),
        ('BH', 'Bahrain'),
        ('BD', 'Bangladesh'),
        ('BB', 'Barbados'),
        ('BY', 'Belarus'),
        ('BE', 'Belgium'),
        ('BZ', 'Belize'),
        ('BJ', 'Benin'),
        ('BM', 'Bermuda'),
        ('BT', 'Bhutan'),
        ('BO', 'Bolivia'),
        ('BA', 'Bosnia and Herzegovina'),
        ('BW', 'Botswana'),
        ('BV', 'Bouvet Island'),
        ('BR', 'Brazil'),
        ('IO', 'British Indian Ocean Territory'),
        ('VG', 'British Virgin Islands'),
        ('BN', 'Brunei'),
        ('BG', 'Bulgaria'),
        ('BF', 'Burkina Faso'),
        ('BI', 'Burundi'),
        ('KH', 'Cambodia'),
        ('CM', 'Cameroon'),
        ('CA', 'Canada'),
        ('CV', 'Cape Verde'),
        ('KY', 'Cayman Islands'),
        ('CF', 'Central African Republic'),
        ('EA', 'Ceuta and Melilla'),
        ('TD', 'Chad'),
        ('CL', 'Chile'),
        ('CN', 'China'),
        ('CX', 'Christmas Island'),
        ('CC', 'Cocos Islands'),
        ('CO', 'Colombia'),
        ('KM', 'Comoros'),
        ('CK', 'Cook Islands'),
        ('CR', 'Costa Rica'),
        ('HR', 'Croatia'),
        ('CU', 'Cuba'),
        ('CW', 'Curaçao'),
        ('CY', 'Cyprus'),
        ('CZ', 'Czech Republic'),
        ('CD', 'Democratic Republic of the Congo'),
        ('DK', 'Denmark'),
        ('DJ', 'Djibouti'),
        ('DM', 'Dominica'),
        ('DO', 'Dominican Republic'),
        ('EC', 'Ecuador'),
        ('EG', 'Egypt'),
        ('SV', 'El Salvador'),
        ('GQ', 'Equatorial Guinea'),
        ('ER', 'Eritrea'),
        ('EE', 'Estonia'),
        ('SZ', 'Eswatini'),
        ('ET', 'Ethiopia'),
        ('FK', 'Falkland Islands'),
        ('FO', 'Faroe Islands'),
        ('FJ', 'Fiji'),
        ('FI', 'Finland'),
        ('FR', 'France'),
        ('GF', 'French Guiana'),
        ('PF', 'French Polynesia'),
        ('TF', 'French Southern Territories'),
        ('GA', 'Gabon'),
        ('GM', 'Gambia'),
        ('GE', 'Georgia'),
        ('DE', 'Germany'),
        ('GH', 'Ghana'),
        ('GI', 'Gibraltar'),
        ('GR', 'Greece'),
        ('GL', 'Greenland'),
        ('GD', 'Grenada'),
        ('GP', 'Guadeloupe'),
        ('GU', 'Guam'),
        ('GT', 'Guatemala'),
        ('GG', 'Guernsey'),
        ('GN', 'Guinea'),
        ('GW', 'Guinea-Bissau'),
        ('GY', 'Guyana'),
        ('HT', 'Haiti'),
        ('HM', 'Heard Island and McDonald Islands'),
        ('HN', 'Honduras'),
        ('HK', 'Hong Kong'),
        ('HU', 'Hungary'),
        ('IS', 'Iceland'),
        ('IN', 'India'),
        ('ID', 'Indonesia'),
        ('IR', 'Iran'),
        ('IQ', 'Iraq'),
        ('IE', 'Ireland'),
        ('IM', 'Isle of Man'),
        ('IL', 'Israel'),
        ('IT', 'Italy'),
        ('JM', 'Jamaica'),
        ('JP', 'Japan'),
        ('JE', 'Jersey'),
        ('JO', 'Jordan'),
        ('KZ', 'Kazakhstan'),
        ('KE', 'Kenya'),
        ('KI', 'Kiribati'),
        ('XK', 'Kosovo'),
        ('KW', 'Kuwait'),
        ('KG', 'Kyrgyzstan'),
        ('LA', 'Laos'),
        ('LV', 'Latvia'),
        ('LB', 'Lebanon'),
        ('LS', 'Lesotho'),
        ('LR', 'Liberia'),
        ('LY', 'Libya'),
        ('LI', 'Liechtenstein'),
        ('LT', 'Lithuania'),
        ('LU', 'Luxembourg'),
        ('MO', 'Macau'),
        ('MK', 'North Macedonia'),
        ('MG', 'Madagascar'),
        ('MW', 'Malawi'),
        ('MY', 'Malaysia'),
        ('MV', 'Maldives'),
        ('ML', 'Mali'),
        ('MT', 'Malta'),
        ('MH', 'Marshall Islands'),
        ('MR', 'Mauritania'),
        ('MU', 'Mauritius'),
        ('YT', 'Mayotte'),
        ('MX', 'Mexico'),
        ('FM', 'Micronesia'),
        ('MD', 'Moldova'),
        ('MC', 'Monaco'),
        ('MN', 'Mongolia'),
        ('ME', 'Montenegro'),
        ('MS', 'Montserrat'),
        ('MA', 'Morocco'),
        ('MZ', 'Mozambique'),
        ('MM', 'Myanmar'),
        ('NA', 'Namibia'),
        ('NR', 'Nauru'),
        ('NP', 'Nepal'),
        ('NL', 'Netherlands'),
        ('NC', 'New Caledonia'),
        ('NZ', 'New Zealand'),
        ('NI', 'Nicaragua'),
        ('NE', 'Niger'),
        ('NG', 'Nigeria'),
        ('NU', 'Niue'),
        ('NF', 'Norfolk Island'),
        ('KP', 'North Korea'),
        ('MP', 'Northern Mariana Islands'),
        ('NO', 'Norway'),
        ('OM', 'Oman'),
        ('PK', 'Pakistan'),
        ('PW', 'Palau'),
        ('PS', 'Palestine'),
        ('PA', 'Panama'),
        ('PG', 'Papua New Guinea'),
        ('PY', 'Paraguay'),
        ('PE', 'Peru'),
        ('PH', 'Philippines'),
        ('PN', 'Pitcairn Islands'),
        ('PL', 'Poland'),
        ('PT', 'Portugal'),
        ('PR', 'Puerto Rico'),
        ('QA', 'Qatar'),
        ('CG', 'Republic of the Congo'),
        ('RO', 'Romania'),
        ('RU', 'Russia'),
        ('RW', 'Rwanda'),
        ('RE', 'Réunion'),
        ('BL', 'Saint Barthélemy'),
        ('SH', 'Saint Helena'),
        ('KN', 'Saint Kitts and Nevis'),
        ('LC', 'Saint Lucia'),
        ('MF', 'Saint Martin'),
        ('PM', 'Saint Pierre and Miquelon'),
        ('VC', 'Saint Vincent and the Grenadines'),
        ('WS', 'Samoa'),
        ('SM', 'San Marino'),
        ('ST', 'Sao Tome and Principe'),
        ('SA', 'Saudi Arabia'),
        ('SN', 'Senegal'),
        ('RS', 'Serbia'),
        ('SC', 'Seychelles'),
        ('SL', 'Sierra Leone'),
        ('SG', 'Singapore'),
        ('SX', 'Sint Maarten'),
        ('SK', 'Slovakia'),
        ('SI', 'Slovenia'),
        ('SB', 'Solomon Islands'),
        ('SO', 'Somalia'),
        ('ZA', 'South Africa'),
        ('GS', 'South Georgia and the South Sandwich Islands'),
        ('KR', 'South Korea'),
        ('SS', 'South Sudan'),
        ('ES', 'Spain'),
        ('LK', 'Sri Lanka'),
        ('SD', 'Sudan'),
        ('SR', 'Suriname'),
        ('SJ', 'Svalbard and Jan Mayen'),
        ('SE', 'Sweden'),
        ('CH', 'Switzerland'),
        ('SY', 'Syria'),
        ('TW', 'Taiwan'),
        ('TJ', 'Tajikistan'),
        ('TZ', 'Tanzania'),
        ('TH', 'Thailand'),
        ('TL', 'Timor-Leste'),
        ('TG', 'Togo'),
        ('TK', 'Tokelau'),
        ('TO', 'Tonga'),
        ('TT', 'Trinidad and Tobago'),
        ('TN', 'Tunisia'),
        ('TR', 'Turkey'),
        ('TM', 'Turkmenistan'),
        ('TC', 'Turks and Caicos Islands'),
        ('TV', 'Tuvalu'),
        ('UG', 'Uganda'),
        ('UA', 'Ukraine'),
        ('AE', 'United Arab Emirates'),
        ('GB', 'United Kingdom'),
        ('US', 'United States'),
        ('UM', 'United States Minor Outlying Islands'),
        ('VI', 'United States Virgin Islands'),
        ('UY', 'Uruguay'),
        ('UZ', 'Uzbekistan'),
        ('VU', 'Vanuatu'),
        ('VA', 'Vatican City'),
        ('VE', 'Venezuela'),
        ('VN', 'Vietnam'),
        ('WF', 'Wallis and Futuna'),
        ('EH', 'Western Sahara'),
        ('YE', 'Yemen'),
        ('ZM', 'Zambia'),
        ('ZW', 'Zimbabwe'),
        ('AX', 'Åland Islands')
    ]



doc = """
Your app description
"""

df_UBI = pd.read_json("survey/UBI_messages.json").T
#df_UBI.columns = df_UBI.columns.str.strip()

df_penalty = pd.read_json("survey/penalty_messages.json").T

df_weight_loss = pd.read_json("survey/weight_loss_messages.json").T


TOPIC_DFS = {
    'UBI': df_UBI,
    'penalty': df_penalty,
    'weight_loss': df_weight_loss,
}


class C(BaseConstants):
    NAME_IN_URL = "survey"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


def creating_session(subsession: Subsession):
    treatments = itertools.cycle(list(product(["package1", "package2", "package3"], repeat=3)))
    for p in subsession.get_players():
        a,b,c = next(treatments)
        p.package_UBI = a
        p.package_penalty = b
        p.package_weight_loss = c

        topic_order = random.choice(list(itertools.permutations(["UBI", "penalty", "weight_loss"])))
        p.first_topic = topic_order[0]
        p.second_topic = topic_order[1]
        p.third_topic = topic_order[2]

        p.message_order_UBI = random.choice(list(range(0,6)))

        p.message_order_penalty = random.choice(list(range(0,6)))

        p.message_order_weight_loss = random.choice(list(range(0,6)))

        p.UBI_statement_formulation = random.choice(['Everyone should receive Universal Basic Income.', 'There should be no Universal Basic Income.'])

        p.penalty_statement_formulation = random.choice(['Penalty shootouts are a good way to determine the winners of football matches.', 'Penalty shootouts are a bad way to determine the winners of football matches.'])

        p.weight_loss_statement_formulation = random.choice(['Using weight loss drugs like Ozempic is a good way to lose weight.', 'Using weight loss drugs like Ozempic is a bad way to lose weight.'])
    


class Group(BaseGroup):
    pass


class Player(BasePlayer):

    #----------------------------------------------------------------------------------------------
    #     TREATMENTS
    #----------------------------------------------------------------------------------------------

    package_UBI = models.StringField()

    package_penalty = models.StringField()

    package_weight_loss = models.StringField()

    first_topic = models.StringField()

    second_topic = models.StringField()

    third_topic = models.StringField()

    message_order_UBI = models.IntegerField()

    message_order_penalty = models.IntegerField()

    message_order_weight_loss = models.IntegerField()

    UBI_statement_formulation = models.StringField()

    penalty_statement_formulation = models.StringField()

    weight_loss_statement_formulation = models.StringField()

    #----------------------------------------------------------------------------------------------
    #     CONSENT
    #----------------------------------------------------------------------------------------------

    consent = models.IntegerField(
        label="I confirm that I have read the description of this study and that I give my consent to participate.",
        choices=[[1, "Yes"],
                 [0, "No"]],
        widget=widgets.RadioSelect,
        blank=False,
    )

    #----------------------------------------------------------------------------------------------
    #     PERSONALITY
    #----------------------------------------------------------------------------------------------


    big1 = models.IntegerField(
        label="I see myself as someone who is reserved",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big2 = models.IntegerField(
        label="I see myself as someone who is generally trusting",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big3 = models.IntegerField(
        label="I see myself as someone who tends to be lazy",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big4 = models.IntegerField(
        label="I see myself as someone who is relaxed, handles stress well",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big5 = models.IntegerField(
        label="I see myself as someone who has few artistic interests",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big6 = models.IntegerField(
        label="I see myself as someone who is outgoing, sociable",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big7 = models.IntegerField(
        label="I see myself as someone who tends to find fault with others",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big8 = models.IntegerField(
        label="I see myself as someone who does a thorough job",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big9 = models.IntegerField(
        label="I see myself as someone who gets nervous easily",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    big10 = models.IntegerField(
        label="I see myself as someone who has an active imagination",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )

    age = models.IntegerField(
        label = "How old are you?",
    )

    gender = models.IntegerField(
        label = "What is your gender?",
        choices = [[0, "female"], [1,"male"], [2, "diverse"]],
    )

    ethnicity = models.IntegerField(
        label = "What is your ethnic background?",
        choices = [[0, "White"],
                   [1, "Asian"],
                   [2, "Black / African descent"],
                   [3, "Hispanic or Latino"],
                   [4, "Multiple ethnic groups"],
                   [5, "Middle Eastern / North African"],
                   [6, "Other ethnic group"]]
    )

    country_of_birth = models.StringField(
        label = "What is your country of birth?",
        choices = country_choices
    )

    country_of_residency = models.StringField(
        label = "What is your country of residency?",
        choices = country_choices
    )

    student_status = models.IntegerField(
        label = "What is your current student status?",
        choices = [[1, "Full-time student"], [2, "Part-time student"], [3, "Not currently a student"]]
    )

    employment_status = models.IntegerField(
        label = "What is your current employment status?",
        choices = [[0, "Employed full-time"],
                   [1, "Employed part-time"],
                   [2, "Self-employed"],
                   [3, "Unemployed"],
                   [4, "Not in the labor force (e.g. retired, caregiver, unable to work)"]]
    )

    occupation_field = models.StringField(
        label= "Please state your field of occupation",
    )

    highest_qualification = models.IntegerField(
        label = "Please state your highest qualification",
        choices = [
            [1, "No formal education"],
            [2, "Secondary school (GCSEs/ O Levels)"],
            [3, "Further education (A Levels, BTEC, apprenticeship)"],
            [4, "Bachelor's degree"],
            [5, "Master's degree"],
            [6, "PhD or higher"]
        ]
    )

    #----------------------------------------------------------------------------------------------
    #     INITIAL AND END BELIEFS
    #----------------------------------------------------------------------------------------------
    
    init_belief_UBI = models.IntegerField(
        label = "Rate your belief on the statement:",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    init_belief_penalty = models.IntegerField(
        label = "Rate your belief on the statement:",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    init_belief_weight_loss = models.IntegerField(
        label = "Rate your belief on the statement:",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    end_belief_UBI = models.IntegerField(
        label = "Rate your belief on the statement:",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    end_belief_penalty = models.IntegerField(
        label = "Rate your belief on the statement:",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    end_belief_weight_loss = models.IntegerField(
        label = "Rate your belief on the statement:",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )

    #----------------------------------------------------------------------------------------------
    #     RANKING OF MESSAGES
    #----------------------------------------------------------------------------------------------

    rank_topic_1_1 = models.IntegerField(
        choices=[1, 2, 3])
    
    rank_topic_1_2 = models.IntegerField(
        choices=[1, 2, 3]
    )
    rank_topic_1_3 = models.IntegerField(
        choices=[1, 2, 3]
    )
    rank_topic_2_1 = models.IntegerField(
        choices=[1, 2, 3])
    
    rank_topic_2_2 = models.IntegerField(
        choices=[1, 2, 3]
    )
    rank_topic_2_3 = models.IntegerField(
        choices=[1, 2, 3]
    )
    rank_topic_3_1 = models.IntegerField(
        choices=[1, 2, 3])
    
    rank_topic_3_2 = models.IntegerField(
        choices=[1, 2, 3]
    )
    rank_topic_3_3 = models.IntegerField(
        choices=[1, 2, 3]
    )

    text_field_UBI = models.LongStringField(label = "Why did you change or retain your belief?")
    text_field_penalty = models.LongStringField(label = "Why did you change or retain your belief?")
    text_field_weight_loss = models.LongStringField(label = "Why did you change or retain your belief?")

    #----------------------------------------------------------------------------------------------
    #     FAMILIARITY AND SECOND ORDER BELIEFS
    #----------------------------------------------------------------------------------------------

    familiarity_UBI = models.IntegerField(
        label = "How familiar are you with the topic?",
        choices=[
            [-2, "Never heard of it"],
            [-1, "Heard of it"],
            [0, "Somewhat familiar"],
            [1, "Familiar"],
            [2, "Very familiar"],
        ],
        widget=widgets.RadioSelect,
    )
    familiarity_penalty = models.IntegerField(
        label = "How familiar are you with the topic?",
        choices=[
            [-2, "Never heard of it"],
            [-1, "Heard of it"],
            [0, "Somewhat familiar"],
            [1, "Familiar"],
            [2, "Very familiar"],
        ],
        widget=widgets.RadioSelect,
    )
    familiarity_weight_loss = models.IntegerField(
        label = "How familiar are you with the topic?",
        choices=[
            [-2, "Never heard of it"],
            [-1, "Heard of it"],
            [0, "Somewhat familiar"],
            [1, "Familiar"],
            [2, "Very familiar"],
        ],
        widget=widgets.RadioSelect,
    )
    second_order_belief_UBI = models.IntegerField(
        label = "What do you belief the average stance of the public is on this statement?",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    second_order_belief_penalty = models.IntegerField(
        label = "What do you belief the average stance of the public is on this statement?",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )
    second_order_belief_weight_loss = models.IntegerField(
        label = "What do you belief the average stance of the public is on this statement?",
        choices=[
            [-2, "Disagree strongly"],
            [-1, "Disagree a little"],
            [0, "Neither agree nor disagree"],
            [1, "Agree a little"],
            [2, "Agree strongly"],
        ],
        widget=widgets.RadioSelect,
    )

    #----------------------------------------------------------------------------------------------
    #     SANITY CHECK MESSAGES (PRO/CON?)
    #----------------------------------------------------------------------------------------------

    stance_topic_1_message_1 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )
    stance_topic_1_message_2 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )
    stance_topic_1_message_3 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )

    stance_topic_2_message_1 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )
    stance_topic_2_message_2 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )
    stance_topic_2_message_3 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )
    
    stance_topic_3_message_1 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )
    stance_topic_3_message_2 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )
    stance_topic_3_message_3 = models.IntegerField(
        label = "Based on the message above, does the author agree or disagree with the initial statement?",
        choices=[
            [1, 'Agree'],
            [0, 'Neither agree nor disagree'],
            [-1, 'Disagree'],
        ],
        )

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

#     PAGES

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

class Transition(Page):
    @staticmethod
    def is_displayed(player):
        # Optional: wenn du willst, dass sie immer gezeigt wird
        return True

class Consent(Page):
    form_model = "player"
    form_fields = ["consent"]

class NoConsent(Page):
    form_model = 'player'
    form_fields = ['consent']

    @staticmethod
    def is_displayed(player):
        return player.consent == 0


class Big5Page(Page):
    form_model = "player"
    form_fields = [
        "age",
        "gender",
        "ethnicity",
        "country_of_birth",
        "country_of_residency",
        "student_status",
        "highest_qualification",
        "employment_status",
        "occupation_field",
        "big1",
        "big2",
        "big3",
        "big4",
        "big5",
        "big6",
        "big7",
        "big8",
        "big9",
        "big10",
    ]

#----------------------------------------------------------------------------------------------
#     INITIAL BELIEF PAGES
#----------------------------------------------------------------------------------------------


class InitBeliefOne(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.first_topic

        return [
            f"init_belief_{topic}",
            f"familiarity_{topic}",
        ]

    @staticmethod
    def vars_for_template(player):
        topic = player.first_topic

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )

class InitBeliefTwo(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.second_topic

        return [
            f"init_belief_{topic}",
            f"familiarity_{topic}",
        ]

    @staticmethod
    def vars_for_template(player):
        topic = player.second_topic

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )

class InitBeliefThree(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.third_topic

        return [
            f"init_belief_{topic}",
            f"familiarity_{topic}",
        ]

    @staticmethod
    def vars_for_template(player):
        topic = player.third_topic

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )
#----------------------------------------------------------------------------------------------
#     MESSAGE PAGES
#----------------------------------------------------------------------------------------------

class MessagePageOne(Page):
    form_model = "player"
    form_fields = ["stance_topic_1_message_1", "stance_topic_1_message_2", "stance_topic_1_message_3"]

    @staticmethod
    def vars_for_template(player):
        topic = player.first_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )

class MessagePageTwo(Page):
    form_model = "player"
    form_fields = ["stance_topic_2_message_1", "stance_topic_2_message_2", "stance_topic_2_message_3"]

    @staticmethod
    def vars_for_template(player):
        topic = player.second_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )

class MessagePageThree(Page):
    form_model = "player"
    form_fields = ["stance_topic_3_message_1", "stance_topic_3_message_2", "stance_topic_3_message_3"]

    @staticmethod
    def vars_for_template(player):
        topic = player.third_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )
    
#----------------------------------------------------------------------------------------------
#     END BELIEF PAGES
#----------------------------------------------------------------------------------------------

class EndBeliefOne(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.first_topic

        return [
            f"end_belief_{topic}",
            f"text_field_{topic}",
        ]
    
    @staticmethod
    def error_message(player, values):
        topic = player.first_topic
        text_value = values.get(f"text_field_{topic}", "")

        if len(text_value.strip()) < 50:
            return "Please enter at least 50 characters in the text field."

    @staticmethod
    def vars_for_template(player):
        topic = player.first_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
        )
    
class EndBeliefTwo(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.second_topic

        return [
            f"end_belief_{topic}",
            f"text_field_{topic}",
        ]
    
    @staticmethod
    def error_message(player, values):
        topic = player.second_topic
        text_value = values.get(f"text_field_{topic}", "")

        if len(text_value.strip()) < 50:
            return "Please enter at least 50 characters in the text field."

    @staticmethod
    def vars_for_template(player):
        topic = player.second_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
        )

class EndBeliefThree(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.third_topic

        return [
            f"end_belief_{topic}",
            f"text_field_{topic}",
        ]
    
    @staticmethod
    def error_message(player, values):
        topic = player.third_topic
        text_value = values.get(f"text_field_{topic}", "")

        if len(text_value.strip()) < 50:
            return "Please enter at least 50 characters in the text field."

    @staticmethod
    def vars_for_template(player):
        topic = player.third_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
        )
    
    def explanation_error_message(self, value):
        if len(value.strip()) < 100:
            return 'Please enter at least 100 characters.'

#----------------------------------------------------------------------------------------------
#     SECOND ORDER BELIEF PAGES
#----------------------------------------------------------------------------------------------

class SecondOrderBeliefOne(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.first_topic

        return [
            f"second_order_belief_{topic}",
        ]

    @staticmethod
    def vars_for_template(player):
        topic = player.first_topic

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )

class SecondOrderBeliefTwo(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.second_topic

        return [
            f"second_order_belief_{topic}",
        ]

    @staticmethod
    def vars_for_template(player):
        topic = player.second_topic

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )

class SecondOrderBeliefThree(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        topic = player.third_topic

        return [
            f"second_order_belief_{topic}",
        ]

    @staticmethod
    def vars_for_template(player):
        topic = player.third_topic

        return dict(
            topic=topic,
            formulation=getattr(player, f"{topic}_statement_formulation"),
            topic_label=topic.replace('_', ' ').upper(),
        )

#----------------------------------------------------------------------------------------------
#     MESSAGE RANKING PAGES
#----------------------------------------------------------------------------------------------

class RankingOne(Page):
    form_model = 'player'
    form_fields = ['rank_topic_1_1', 'rank_topic_1_2', 'rank_topic_1_3']

    @staticmethod
    def vars_for_template(player):
        topic = player.first_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
        )

    def error_message(self, values):
        ranks = [values['rank_topic_1_1'], values['rank_topic_1_2'], values['rank_topic_1_3']]
        if len(set(ranks)) < 3:
            return "Each message must have a unique rank (1, 2, 3)."
           
class RankingTwo(Page):
    form_model = 'player'
    form_fields = ['rank_topic_2_1', 'rank_topic_2_2', 'rank_topic_2_3']

    @staticmethod
    def vars_for_template(player):
        topic = player.second_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
        )

    def error_message(self, values):
        ranks = [values['rank_topic_2_1'], values['rank_topic_2_2'], values['rank_topic_2_3']]
        if len(set(ranks)) < 3:
            return "Each message must have a unique rank (1, 2, 3)."

class RankingThree(Page):
    form_model = 'player'
    form_fields = ['rank_topic_3_1', 'rank_topic_3_2', 'rank_topic_3_3']

    @staticmethod
    def vars_for_template(player):
        topic = player.third_topic

        perms = list(itertools.permutations([1, 2, 3]))

        message_order = getattr(player, f"message_order_{topic}")
        package = getattr(player, f"package_{topic}")

        df = TOPIC_DFS[topic]

        return dict(
            topic=topic,
            message1=df.loc[perms[message_order][0], package],
            message2=df.loc[perms[message_order][1], package],
            message3=df.loc[perms[message_order][2], package],
        )

    def error_message(self, values):
        ranks = [values['rank_topic_3_1'], values['rank_topic_3_2'], values['rank_topic_3_3']]
        if len(set(ranks)) < 3:
            return "Each message must have a unique rank (1, 2, 3)."

#----------------------------------------------------------------------------------------------
#     THANK YOU PAGE
#----------------------------------------------------------------------------------------------

class ThankYou(Page):
    @staticmethod
    def is_displayed(player):
        # Optional: wenn du willst, dass sie immer gezeigt wird
        return True
    


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

#     PAGE SEQUENCE

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

introduction = [Consent, NoConsent]

topic_one_pages= [InitBeliefOne, MessagePageOne, EndBeliefOne, RankingOne, SecondOrderBeliefOne]

transitions_page = [Transition]

topic_two_pages = [InitBeliefTwo, MessagePageTwo, EndBeliefTwo, RankingTwo, SecondOrderBeliefTwo]

transitions_page = [Transition]

topic_three_pages = [InitBeliefThree, MessagePageThree, EndBeliefThree, RankingThree, SecondOrderBeliefThree]

ending = [Big5Page, ThankYou]

#page_sequence = [MessagePageOne,MessagePageTwo, MessagePageThree]

page_sequence = introduction + topic_one_pages + transitions_page + topic_two_pages + transitions_page + topic_three_pages + ending

