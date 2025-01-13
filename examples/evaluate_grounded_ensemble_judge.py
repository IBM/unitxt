from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict

data = {
    "test": [
        {
            "conversation": "user: is eugene from walking dead really a scientist\nagent: No, Eugene from The Walking Dead is not a real scientist. He lied about being a scientist to manipulate the other survivors into taking him to Washington D.C., believing it to be the best chance for survival. In reality, he is a high school science teacher who does not know how to cure the virus. However, he is shown to be highly intelligent and resourceful, able to do things such as repairing the solar power system in Alexandria and making bullets for the Saviors.\nuser: Why did he lied about himself being a scientist?\nagent: Eugene lied about being a scientist to manipulate the other survivors into taking him to Washington D.C., believing it to be the best chance for survival. He initially used this lie to gain the protection of Abraham Ford. He assists the group with technical tasks even after the truth is revealed, which also improves his own confidence. His lie caused tension within the group, particularly between him and Abraham, but eventually Abraham forgave him and they resumed their friendship.\nuser: Is The Walking Dead currently shown on TV? Which station?",
            "document": "The Walking Dead (season 6)\nOn October 7 , 2014 , AMC renewed The Walking Dead for a sixth season . Scott M. Gimple said that the sixth season would continue to remix material from the comic and explained that there would be a flashback backstory to some of the characters :\n\nThe Walking Dead (season 6)\nThe sixth season of The Walking Dead , an American post-apocalyptic horror television series on AMC , premiered on October 11 , 2015 , and concluded on April 3 , 2016 , consisting of 16 episodes . Developed for television by Frank Darabont , the series is based on the eponymous series of comic books by Robert Kirkman , Tony Moore , and Charlie Adlard . The executive producers are Kirkman , David Alpert , Scott M. Gimple , Greg Nicotero , Tom Luse , and Gale Anne Hurd , with Gimple as showrunner for the third consecutive season . The sixth season received positive reviews by critics . It was nominated for multiple awards and won three , including Best Horror Television Series , at the 42nd Saturn Awards .",
        },
        {
            "conversation": "user: is eugene from walking dead really a scientist\nagent: No, Eugene from The Walking Dead is not a real scientist. He lied about being a scientist to manipulate the other survivors into taking him to Washington D.C., believing it to be the best chance for survival. In reality, he is a high school science teacher who does not know how to cure the virus. However, he is shown to be highly intelligent and resourceful, able to do things such as repairing the solar power system in Alexandria and making bullets for the Saviors.\nuser: Why did he lied about himself being a scientist?\nagent: Eugene lied about being a scientist to manipulate the other survivors into taking him to Washington D.C., believing it to be the best chance for survival. He initially used this lie to gain the protection of Abraham Ford. He assists the group with technical tasks even after the truth is revealed, which also improves his own confidence. His lie caused tension within the group, particularly between him and Abraham, but eventually Abraham forgave him and they resumed their friendship.\nuser: Is The Walking Dead currently shown on TV? Which station?",
            "document": "The Walking Dead (season 6)\nOn October 7 , 2014 , AMC renewed The Walking Dead for a sixth season . Scott M. Gimple said that the sixth season would continue to remix material from the comic and explained that there would be a flashback backstory to some of the characters :\n\nThe Walking Dead (season 6)\nThe sixth season of The Walking Dead , an American post-apocalyptic horror television series on AMC , premiered on October 11 , 2015 , and concluded on April 3 , 2016 , consisting of 16 episodes . Developed for television by Frank Darabont , the series is based on the eponymous series of comic books by Robert Kirkman , Tony Moore , and Charlie Adlard . The executive producers are Kirkman , David Alpert , Scott M. Gimple , Greg Nicotero , Tom Luse , and Gale Anne Hurd , with Gimple as showrunner for the third consecutive season . The sixth season received positive reviews by critics . It was nominated for multiple awards and won three , including Best Horror Television Series , at the 42nd Saturn Awards .",
        },
        {
            "conversation": "user: is eugene from walking dead really a scientist\nagent: No, Eugene from The Walking Dead is not a real scientist. He lied about being a scientist to manipulate the other survivors into taking him to Washington D.C., believing it to be the best chance for survival. In reality, he is a high school science teacher who does not know how to cure the virus. However, he is shown to be highly intelligent and resourceful, able to do things such as repairing the solar power system in Alexandria and making bullets for the Saviors.\nuser: Why did he lied about himself being a scientist?",
            "document": "Eugene Porter\nDr. Eugene Porter is a fictional character from the comic book series The Walking Dead and is portrayed by Josh McDermitt beginning in the fourth season of the American television series of the same name . In both mediums he claims to be a scientist who knows the cure to the zombie plague and is being escorted to Washington D.C. by Sgt . Abraham Ford and Rosita Espinosa , and encounter Rick Grimes and his group and recruit them to assist their mission . Eugene is overweight , possesses virtually no survival skills of his own and is extremely dependent on the group , but is highly intelligent and resourceful in using technology to ensure the group 's survival . Eventually it is revealed that Eugene is not a scientist , but a high school science teacher , and that he does n't know how to cure the virus but lied to manipulate the other survivors into taking him to Washington D.C. believing it to be the best chance for survival . This proves true as the group eventually finds the Alexandria Safe - Zone where Eugene becomes its primary engineer . Though his lie puts a strain on their friendship , Abraham eventually forgives him and they resume being friends .\n\nList of The Walking Dead (TV series) characters\nDr. Eugene Porter , portrayed by Josh McDermitt , is a survivor of the apocalypse and a member of Abraham Ford 's survivor group and later a minor antagonist after he unwillingly begins working for the Saviors . He comes across as timid and socially awkward . Initially when encountered by Rick 's group , he appears to be a scientist that knows of a cure to the outbreak located in Washington , D.C. , with Abraham and Rosita assisting him . Later , Eugene is forced to reveal that he lied to Abraham to gain his protection back when they first met , leading Abraham to punch him unconscious and creating a rift in their friendship . When the group makes it to Alexandria , Eugene assists the community with his technical knowledge to repair their solar power system . This helps to boost his confidence to learn how to defend himself , reconcile his friendship with Abraham , and help the community find a way to make more bullets . When Negan discovers that Eugene makes bullets , Eugene is taken to work for the Saviors . Although Eugene accepts the offer to become one of the Saviors , he still helps Sasha kill herself but lies about it to Negan . He is an important player for the Saviors in the All Out War arc , but is tormented with conflicting loyalties , especially after he discovers that Dwight is playing both sides .",
        },
    ]
}

predictions = [
    "No, The Walking Dead is not currently shown on TV. The final episode of the series aired on April 17, 2022, on AMC. The show has completed its 11-season run. However, if you're interested in watching previous seasons, they are available to stream on various platforms such as AMC+, Hulu, and Peacock. Additionally, you can find The Walking Dead merchandise and spin-off series on these platforms and other marketplaces.",
    "Yes, The Walking Dead is currently being shown on TV. It airs on AMC, an American basic cable network. AMV airs new episodes of The Walking Dead as well as reruns of previous seasons.",
    "Eugene lied about being a scientist to manipulate the other survivors into taking him to Washington D.C., believing it to be the best chance for survival. He initially used this lie to gain the protection of Abraham Ford. He assists the group with technical tasks even after the truth is revealed, which also improves his own confidence. His lie caused tension within the group, particularly between him and Abraham, but eventually Abraham forgave him and they resumed their friendship.",
]


card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        input_fields={"conversation": "str", "document": "str"},
        reference_fields={},
        prediction_type="str",
        metrics=[
            "metrics.llm_as_judge.conversation_answer_groundedness.ensemble_v1_wml_judges"
        ],
    ),
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                input_format="{conversation}\n\n\nDocument:\n{document}",
                output_format="",
            )
        }
    ),
)

dataset = load_dataset(card=card, template_card_index="simple", split="test")
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)
