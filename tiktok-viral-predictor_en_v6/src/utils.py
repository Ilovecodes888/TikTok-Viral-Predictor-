
import re
EMOJI_PATTERN = re.compile('['
    '\U0001F600-\U0001F64F'
    '\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF'
    '\U0001F1E0-\U0001F1FF'
']+')
def count_emojis(text: str) -> int:
    return len(EMOJI_PATTERN.findall(text or ""))
