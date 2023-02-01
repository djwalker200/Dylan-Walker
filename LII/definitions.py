class DefinedTerm:

    def __init__(self,
                 term,
                 definition,
                 idnum,
                 scope=None,
                 default_scope=False):
        self.term = term
        self.definition = definition
        self.idnum = idnum
        self.scope = scope
        self.n_occur = 0
        self.default_scope = default_scope
        self.ID = self.generate_id()
        self.scopeID = self.generate_scope_id()

    def generate_id(self):
        return hash((self.term, self.idnum))

    def generate_scope_id(self):
        return hash((self.term, self.scope, self.idnum))

    def __repr__(self):
        return f"Found Term: {self.term} \n in file {self.idnum} \n scope {self.scope} \n default_scope {self.default_scope} \n definition {self.definition} {self.ID} \n nOccur {self.n_occur} \n "

    def __str__(self):

        return f"Found Term: {self.term} \n in file {self.idnum} \n scope {self.scope} \n default_scope {self.default_scope} \n definition {self.definition} {self.ID} \n nOccur {self.n_occur} \n "

    def to_list(self):
        return [
            self.term, self.idnum, self.definition, self.ID, self.scope,
            self.scopeID
        ]

    def scopeContains(self, target_location):

        return (target_location.find(self.scope) >= 0)


class FoundTerm:

    def __init__(self, term, start, end, idnum):
        self.term = term
        self.start = start
        self.end = end
        self.idnum = idnum
        self.definition = None
        self.definition_id = None
        self.default_scope = False
        self.n_instance = 0

    def __repr__(self):
        return f"Found Term: {self.term} \n in file {self.idnum} \n start = {self.start} \n end = {self.end} \n definition {self.definition} {self.definition_id} \n scope {self.default_scope} \n n {self.n_instance} \n "

    def __str__(self):

        return f"Found Term: {self.term} \n in file {self.idnum} \n start = {self.start} \n end = {self.end} \n definition {self.definition} {self.definition_id} \n scope {self.default_scope} \n instance number {self.n_instance} \n "

    def assignDefinition(self, candidate):
        self.definition = candidate.definition
        self.definition_id = candidate.ID
        self.default_scope = candidate.default_scope
        candidate.n_occur += 1
        self.n_instance = candidate.n_occur

    def to_list(self):
        return [self.term, self.idnum, self.definition, self.definition_id]
