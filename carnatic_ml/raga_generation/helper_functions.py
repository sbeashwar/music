import random

def create_emphasis_phrase(target_swaras, allowed_swaras, length=4):
        """Create a phrase that emphasizes certain swaras while using allowed notes."""
        phrase = []
        if not isinstance(target_swaras, list):
            target_swaras = [target_swaras]
            
        # Start with a target swara
        phrase.append(random.choice(target_swaras))
        
        # Build phrase of specified length
        while len(phrase) < length:
            if random.random() < 0.6:  # 60% chance to use emphasis note
                phrase.append(random.choice(target_swaras))
            else:
                # Use another allowed swara, preferring adjacent notes
                options = [s for s in allowed_swaras if s not in phrase[-1:]]
                phrase.append(random.choice(options))
        
        return phrase

def adjust_phrase_ending(phrase, nyasa_swaras):
        """Modify a phrase to end on one of the nyasa swaras."""
        if not isinstance(nyasa_swaras, list):
            nyasa_swaras = [nyasa_swaras]
            
        # Copy all but last note
        adjusted = phrase[:-1]
        # End on nyasa swara
        adjusted.append(random.choice(nyasa_swaras))
        return adjusted