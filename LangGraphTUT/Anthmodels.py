from anthropic import Anthropic
from dotenv import load_dotenv; load_dotenv()

client = Anthropic()
print([m.id for m in client.models.list().data])
