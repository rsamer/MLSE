# -*- coding: utf-8 -*-

import logging
import nltk
import os
from nltk.tag.stanford import StanfordPOSTagger
from util import helper

pos_tagger_dir = os.path.join(helper.APP_PATH, 'postagger')
pos_tagger_data_path = os.path.join(pos_tagger_dir, 'models', 'english-bidirectional-distsim.tagger')
pos_tagger_jar_path = os.path.join(pos_tagger_dir, 'stanford-postagger.jar')
nltk.data.path = [os.path.join(helper.APP_PATH, "corpora", "nltk_data")]

_logger = logging.getLogger(__name__)

# TODO: Stanford POS-tagging
def pos_tagging(posts):
    '''
        POS-Tagging via Stanford POS tagger
        NOTE: This library creates a Java process in the background.
              Please make sure you have installed Java 1.6 or higher.

              sudo apt-get install default-jre
              sudo apt-get install default-jdk
    '''
    _logger.info("Pos-tagging for posts' tokens")

    '''
        See: http://www.comp.leeds.ac.uk/ccalas/tagsets/upenn.html
        --------------------------------------------------------------------------------------------
        Tag    Description                         Examples
        --------------------------------------------------------------------------------------------
        CC     conjunction, coordinating           & 'n and both but either et for less minus neither nor or plus so therefore times v. versus vs. whether yet
        CD     numeral, cardinal                   mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025 fifteen 271,124 dozen quintillion DM2,000 ...
        DT     determiner                          all an another any both del each either every half la many much nary neither no some such that the them these this those
        EX     existential there                   there
        FW     foreign word                        gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte terram fiche oui corporis ...
        IN     preposition or conjunction, subordinating astride among uppon whether out inside pro despite on by throughout below within for towards near behind atop around if like until below next into if beside ...
        JJ     adjective or numeral,ordinal        third ill-mannered pre-war regrettable oiled calamitous first separable ectoplasmic battery-powered participatory fourth still-to-be-named multilingual multi-disciplinary ...
        JJR    adjective, comparative              bleaker braver breezier briefer brighter brisker broader bumper busier calmer cheaper choosier cleaner clearer closer colder commoner costlier cozier creamier crunchier cuter ...
        JJS    adjective, superlative              calmest cheapest choicest classiest cleanest clearest closest commonest corniest costliest crassest creepiest crudest cutest darkest deadliest dearest deepest densest dinkiest ...
        LS     list item marker                    A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005 SP-44007 Second Third Three Two \* a b c d first five four one six three two
        MD     modal auxiliary                     can cannot could couldn't dare may might must need ought shall should shouldn't will would
        NN     noun, common, singular or mass      common-carrier cabbage knuckle-duster Casino afghan shed thermostat investment slide humour falloff slick wind hyena override subhumanity machinist ...
        NNP    noun, proper, singular              Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA Shannon A.K.C. Meltex Liverpool ...
        NNPS   noun, proper, plural                Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques Apache Apaches Apocrypha ...
        NNS    noun, common, plural                undergraduates scotches bric-a-brac products bodyguards facets coasts divestitures storehouses designs clubs fragrances averages subjectivists apprehensions muses factory-jobs ...
        PDT    pre-determiner                      all both half many quite such sure this
        POS    genitive marker                     ' 's
        PRP    pronoun, personal                   hers herself him himself hisself it itself me myself one oneself ours ourselves ownself self she thee theirs them themselves they thou thy us
        PRP$   pronoun, possessive                 her his mine my our ours their thy your
        RB     adverb                              occasionally unabatingly maddeningly adventurously professedly stirringly prominently technologically magisterially predominately swiftly fiscally pitilessly ...
        RBR    adverb, comparative                 further gloomier grander graver greater grimmer harder harsher healthier heavier higher however larger later leaner lengthier less-perfectly lesser lonelier longer louder lower more ...
        RBS    adverb, superlative                 best biggest bluntest earliest farthest first furthest hardest heartiest highest largest least less most nearest second tightest worst
        RP     particle                            aboard about across along apart around aside at away back before behind by crop down ever fast for forth from go high i.e. in into just later low more off on open out over per pie raising start teeth that through under unto up up-pp upon whole with you
        TO     "to" as preposition or infinitive marker    to
        UH     interjection                        Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly man baby diddle hush sonuvabitch ...
        VB     verb, base form                     ask assemble assess assign assume atone attention avoid bake balkanize bank begin behold believe bend benefit bevel beware bless boil bomb boost brace break bring broil brush build ...
        VBD    verb, past tense                    dipped pleaded swiped regummed soaked tidied convened halted registered cushioned exacted snubbed strode aimed adopted belied figgered speculated wore appreciated contemplated ...
        VBG    verb, present participle or gerund  telegraphing stirring focusing angering judging stalling lactating hankerin' alleging veering capping approaching traveling besieging encrypting interrupting erasing wincing ...
        VBN    verb, past participle               multihulled dilapidated aerosolized chaired languished panelized used experimented flourished imitated reunifed factored condensed sheared unsettled primed dubbed desired ...
        VBP    verb, present tense, not 3rd person singular    predominate wrap resort sue twist spill cure lengthen brush terminate appear tend stray glisten obtain comprise detest tease attract emphasize mold postpone sever return wag ...
        VBZ    verb, present tense, 3rd person singular  bases reconstructs marks mixes displeases seals carps weaves snatches slumps stretches authorizes smolders pictures emerges stockpiles seduces fizzes uses bolsters slaps speaks pleads ...
        WDT    WH-determiner                       that what whatever which whichever
        WP     WH-pronoun                          that what whatever whatsoever which who whom whosoever
        WP$    WH-pronoun, possessive              whose
        WRB    Wh-adverb                           how however whence whenever where whereby whereever wherein whereof why
    '''
    progress_bar = helper.ProgressBar(len(posts))
    pos_tags_black_list = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
#    pos_tags_black_list = ['CC', 'CD', 'DT', 'EX', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB']
    existing_stanford_pos_tags = set()
    removed_stanford_tokens = set()
    # Note: "-mx30g" sets java's max memory size to 30 GB RAM
    #       Please change when experiencing OS-related problems!
    for post in posts:
        english_postagger = StanfordPOSTagger(pos_tagger_data_path, pos_tagger_jar_path, java_options='-mx30g')
        pos_tagged_tokens = english_postagger.tag(post.tokens)
        tagged_tokens = filter(lambda t: t[1] not in pos_tags_black_list, pos_tagged_tokens)
        post.tokens = map(lambda t: t[0], tagged_tokens)
        post.tokens_pos_tags = map(lambda t: t[1], tagged_tokens)
        removed_stanford_tokens |= set(filter(lambda t: t[1] in pos_tags_black_list, pos_tagged_tokens))
        existing_stanford_pos_tags |= set(map(lambda t: t[1], pos_tagged_tokens))
        progress_bar.update()
    print existing_stanford_pos_tags
    for t in removed_stanford_tokens:
        print t
    print len(posts)
    print len(removed_stanford_tokens)
    print "=" * 80 + "\n\n"
    progress_bar.finish()
    return

#     '''
#         --------------------------------------------------------------------------------------------
#         Tag    Meaning    English Examples
#         --------------------------------------------------------------------------------------------
#         ADJ    adjective    new, good, high, special, big, local
#         ADP    adposition    on, of, at, with, by, into, under
#         ADV    adverb    really, already, still, early, now
#         CONJ    conjunction    and, or, but, if, while, although
#         DET    determiner, article    the, a, some, most, every, no, which
#         NOUN    noun    year, home, costs, time, Africa
#         NUM    numeral    twenty-four, fourth, 1991, 14:24
#         PRT    particle    at, on, out, over per, that, up, with
#         PRON    pronoun    he, their, her, its, my, I, us
#         VERB    verb    is, say, told, given, playing, would
#         .    punctuation marks    . , ; !
#         X    other    ersatz, esprit, dunno, gr8, univeristy
#     '''
#     pos_tags_black_list = ['CONJ', 'DET', 'PRT', 'PRON', '.']
#     existing_pos_tags = set()
#     removed_tokens = set()
#     for post in posts:
#         pos_tagged_tokens = nltk.pos_tag(post.tokens, tagset='universal')
#         tagged_tokens = filter(lambda t: t[1] not in pos_tags_black_list, pos_tagged_tokens)
#         #post.tokens = map(lambda t: t[0], tagged_tokens)
#         post.tokens_pos_tags = map(lambda t: t[1], tagged_tokens)
#         removed_tokens |= set(filter(lambda t: t[1] in pos_tags_black_list, pos_tagged_tokens))
#         existing_pos_tags |= set(map(lambda t: t[1], pos_tagged_tokens))
#     import sys;sys.exit()
#     print "=" * 80 + "\n\n"
#     print existing_pos_tags
#     print removed_tokens
