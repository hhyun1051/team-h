
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.homeassistant_api import HomeAssistantAPIClient

async def list_all_entities():
    load_dotenv()
    
    token = os.getenv("HOMEASSISTANT_TOKEN")
    if not token:
        print("Error: HOMEASSISTANT_TOKEN not found in environment")
        return

    client = HomeAssistantAPIClient(
        url=os.getenv("HOMEASSISTANT_URL", "http://localhost:8124"),
        token=token
    )

    if not await client.health_check():
        print("❌ Home Assistant API connection failed")
        return

    states = await client.get_states()
    print(f"Found {len(states)} entities:")
    
    for state in states:
        print(f"{state.entity_id}: {state.state} ({state.attributes.get('friendly_name', 'No Name')})")

if __name__ == "__main__":
    asyncio.run(list_all_entities())
