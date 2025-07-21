from bedrock_agentcore.memory import MemoryClient

memory_client = MemoryClient(region_name="us-west-2", environment="prod")

memory_id = "test_memory_d382b576-vOpWPB2KeK"

status = memory_client.get_memory_status(memory_id)

print(f"Memory status: {status}")

# event = memory_client.create_event(
#     memory_id=memory_id,
#     actor_id="test_actor",
#     session_id=uuid.uuid4().hex,
#     messages=[
#         ("Hello, what is your name?", "USER"),
#         ("My name is AgentCore.", "ASSISTANT"),
#     ],
# )

# print(f"Created event: {event}")

memories = memory_client.retrieve_memories(memory_id=memory_id, namespace="summaries/test_actor/", query="name")

synopsis = "\n".join([m.get("content", {}).get("text", "") for m in memories])

print(f"Retrieved memories: {synopsis}")
