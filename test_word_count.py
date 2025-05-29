def count_words(text):
    """
    Count words in text more accurately:
    - Handles multiple spaces/newlines
    - Handles punctuation
    - Handles special characters
    - Counts hyphenated words as one word
    """
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Count words, handling special cases
    words = text.split()
    word_count = 0
    
    for word in words:
        # Remove punctuation from word edges
        word = word.strip('.,!?()[]{}:;"\'/\\')
        
        # Skip if empty after cleaning
        if not word:
            continue
            
        # Count numbers as words
        if word.replace('.', '').replace(',', '').isdigit():
            word_count += 1
            continue
            
        # Count hyphenated words as one word
        if '-' in word:
            word_count += 1
            continue
            
        # Count regular words
        if any(c.isalpha() for c in word):
            word_count += 1
            
    return word_count

# Read and count words in the article
file_path = "/Users/christospapaventsis/Desktop/article.txt"
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        word_count = count_words(content)
        print(f"Word count using our function: {word_count}")
        print(f"Estimated duration (minutes): {word_count / 200:.2f}")
        print(f"Estimated duration (seconds): {(word_count / 200) * 60:.2f}")
except Exception as e:
    print(f"Error reading file: {e}") 