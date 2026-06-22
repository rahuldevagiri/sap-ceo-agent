from src.storage.repository import Repository
from src.config import RETRIEVAL_TOP_K

class RAGRetriever:
    def __init__(self):
        self.repo = Repository()

    def retrieve(self, query: str, top_k: int = 5):
        return self.repo.search(query=query, top_k=top_k)

    def retrieve_multi_view(self, company: str = "SAP", top_k: int = RETRIEVAL_TOP_K):
        query_sets = {
            "opportunities": f"{company} business opportunities growth AI cloud partnerships product expansion",
            "risks": f"{company} risks threats regulation competition negative sentiment operational issues",
            "competitors": f"{company} competitors Oracle Salesforce Microsoft Workday ServiceNow market moves",
            "trends": f"{company} enterprise software trends AI cloud ERP automation data platforms"
        }

        results = {}
        for key, query in query_sets.items():
            results[key] = self.retrieve(query=query, top_k=top_k)

        return results