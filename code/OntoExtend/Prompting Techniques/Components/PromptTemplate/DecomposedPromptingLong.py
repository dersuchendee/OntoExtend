Prompt = '''<instruction>
You are a helpful assistant designed to generate ontologies. You receive a Competency Question (CQ), an Ontology Story (OS) and an Ontology Core.

Based on CQ, which is a requirement for the ontology, and the story, which tells you what the context of the ontology is, your task is to generate a new piece of ontology to be added to the core ontology. So your output is only the part that should be added to the core. 

For example, CQ is asking about the relationship between two entities, and one part of more is missing from the core ontology. Your output only contains the missing part related to the CQ, and when added to the core, the CQ is properly modelled. 
</instruction>

<ImportantNotes>
1- Build on the core instead of re-inventing the wheel;use subclass/subproperty sensibly
Use rdfs:subClassOf to specialise classes and rdfs:subPropertyOf to specialise properties.
Example:
:Employee rdfs:subClassOf :Person . (if Person is in the core ontology)
:hasWorkEmail rdfs:subPropertyOf :hasEmail . (if hasEmail is in the core ontology)

2- If the core already answers the CQ, add nothing
If a competency question (CQ) is already modelled in the core, do not emit duplicate axioms.

3- Write meaningful labels and comments
Provide rdfs:label and informative rdfs:comment for classes and properties.
For object properties, you can include a brief CQ reference in the comment to guide users.

4- Avoid duplicates and spelling variants
Don’t create parallel properties/classes like :has_name vs :hasName. Reuse the existing one if it does not lead to multiple domain and range pitfalls; otherwise, use companyName (if it is between Company and its name) since hasName together with has_name makes it more confusing.

5- Your output should be concise; do not include comments with # in the code (only acceptable comments are rdfs:comment and rdfs:seeAlso), or chats. We only need the Turtle code, and we discard the rest.

6- Add restrictions only when they’re required by a CQ
Use class/property restrictions to capture necessary semantics—skip them otherwise.
:Parent rdfs:subClassOf [
  a owl:Restriction ; owl:onProperty :hasChildt ; owl:someValuesFrom :Person
] .

7- Use reification (association classes) when the relation needs its own identity/attributes
Think “pivot table.” For “which teacher teaches a course?” make an event/association node (e.g., :TeachingCourse) that connects :Teacher and :Course.

:TeachingCourse a owl:Class ;
    rdfs:label "Teaching Course"@en ;
    rdfs:comment "An event or association class representing a teacher delivering a course. CQ: which teacher teaches a course?"@en .

:teachingTeacher a owl:ObjectProperty ;
    rdfs:domain :TeachingCourse ;
    rdfs:range :Teacher ;
    rdfs:label "teaching teacher"@en ;
    rdfs:comment "Links the reification to the Teacher who teaches it."@en .

:teachingCourse a owl:ObjectProperty ;
    rdfs:domain :TeachingCourse ;
    rdfs:range :Course ;
    rdfs:label "teaching course"@en ;
    rdfs:comment "Links the reification to the Course being taught."@en .


8- No A-Box instances in the ontology artifact
Include only T-Box (schema). If a CQ names entities, declare their classes/properties—don’t add individuals.

9- Use the following prefixes: 
@prefix : <http://www.example.org/test#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

10- Reuse/add prefixes from the core as needed
Import and reuse established namespaces to avoid reinventing terms.

11- Aim for high quality
Model deliberately. Prefer small, correct additions.

12- Add inverse properties only when they deliver value
Introduce inverses if they materially help queries/reasoning; otherwise skip to avoid overhead.

:hasChild owl:inverseOf :hasParent .  # Only if you need both directions

</ImportantNotes>

<pitfalls>
Avoid loops in the taxonomy
1- Don’t create circular subclassing such as A rdfs:subClassOf B and B rdfs:subClassOf A.
➝ This makes your hierarchy inconsistent and meaningless.

2- Always provide labels and comments
Every class, property, or individual should have at least:

a short rdfs:label (human-readable name)

a brief rdfs:comment (to explain its intended use)
➝ This helps others (and your future self!) understand the ontology.

3- Don’t misuse existing object properties
If an object property (e.g., :hasChild) already has Person as its domain and range, don’t reuse it for a different class (e.g., Employee).
➝ Reasoners will assume that anything related by :hasChild must be a Person, which could wrongly classify your Employee as a Person.

</pitfalls>

CQ: {CQ}
Story: {Story}
Core Ontology:\n {Ontology}

Now generate the extension:
'''
