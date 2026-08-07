import os
import glob

def backup_ai_memory():
    # Find the Gemini brain directory
    brain_dir = os.path.expanduser('~/.gemini/antigravity/brain/')
    
    # The crucial artifacts we need to restore context tomorrow
    artifacts = ['implementation_plan.md', 'task.md', 'walkthrough.md']
    
    # Output file saved directly inside the vision-track project folder
    output_file = os.path.expanduser('~/vision-track/AI_MEMORY_BACKUP.md')
    
    with open(output_file, 'w') as outfile:
        outfile.write("# VISION-TRACK AI MEMORY BACKUP\n\n")
        outfile.write("> **Instructions for Tomorrow:**\n")
        outfile.write("> Just tell the AI: *'Read the AI_MEMORY_BACKUP.md file in the project folder to restore your context, then let's continue with Phase 2.'*\n\n")
        
        # Find the most recently active conversation folder
        all_plans = glob.glob(os.path.join(brain_dir, '*', 'implementation_plan.md'))
        if not all_plans:
            print("No AI artifacts found to backup.")
            return
            
        active_plan = max(all_plans, key=os.path.getmtime)
        active_brain_folder = os.path.dirname(active_plan)
        
        for artifact in artifacts:
            filepath = os.path.join(active_brain_folder, artifact)
            if os.path.exists(filepath):
                outfile.write(f"## --- {artifact.upper()} ---\n\n")
                with open(filepath, 'r') as infile:
                    outfile.write(infile.read())
                outfile.write("\n\n")
                print(f"Successfully backed up: {artifact}")
                
    # --- EXTRACT CONVERSATION RESPONSES ---
    import json
    conversation_id = os.path.basename(active_brain_folder)
    transcript_path = os.path.expanduser(f'~/.gemini/antigravity/brain/{conversation_id}/.system_generated/logs/transcript.jsonl')
    responses_file = os.path.expanduser('~/vision-track/conversation-responses.md')
    
    if os.path.exists(transcript_path):
        with open(responses_file, 'w') as f_out:
            f_out.write("# AI Conversation History\n\n")
            with open(transcript_path, 'r') as f_in:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'PLANNER_RESPONSE' and data.get('content'):
                            f_out.write("### AI Response:\n")
                            f_out.write(data['content'] + "\n\n---\n\n")
                    except:
                        continue
        print(f"Successfully backed up conversation to: {responses_file}")
                
    print(f"\nSUCCESS! AI context saved to: {output_file}")
    print("This file will now be included when you ZIP your vision-track folder!")

if __name__ == '__main__':
    backup_ai_memory()
