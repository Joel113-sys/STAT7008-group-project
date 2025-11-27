def process_vision_info(messages):
    """Extract images from chat messages."""
    image_inputs = []
    video_inputs = []
    for message in messages:
        for item in message["content"]:
            if item.get("type") == "image":
                image_inputs.append(item["image"])
    return image_inputs if image_inputs else None, video_inputs
