#!/usr/bin/env python3
import sys, json, argparse, logging
from agents.orchestrator import Orchestrator
 
logging.basicConfig(
    level=logging.WARNING,
    format='%(name)s %(levelname)s: %(message)s',
)
 
BANNER = '''
                                                                                  
  ▄▄▄▄   ▄▄▄    ▄▄▄ ▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄ ▄▄▄      ▄▄▄   ▄▄▄▄▄   ▄▄▄    ▄▄▄ ▄▄▄   ▄▄▄ 
▄██▀▀██▄ ████▄  ███ ▀▀▀███▀▀▀  ███  ████▄  ▄████ ▄███████▄ ████▄  ███ ███   ███ 
███  ███ ███▀██▄███    ███     ███  ███▀████▀███ ███   ███ ███▀██▄███ ▀███▄███▀ 
███▀▀███ ███  ▀████    ███     ███  ███  ▀▀  ███ ███▄▄▄███ ███  ▀████   ▀███▀   
███  ███ ███    ███    ███    ▄███▄ ███      ███  ▀█████▀  ███    ███    ███    
                                                                                                                                      
  Type /help for commands. Ctrl+C to exit.
'''
 
def fmt_result(r: dict) -> str:
    lines = []
    if r.get('warnings'):
        lines.append(f'  [SAFETY] {r["warnings"]}')
    if r.get('confirmation_q'):
        lines.append(f'\n  [?] {r["confirmation_q"]}\n')
    lines.append(r.get('response', '(no response)'))
    if r.get('low_conf'):
        c = r.get('confidence', 0)
        lines.append(f'\n  [confidence: {c:.0%} — verify from primary sources]')
    if r.get('sources'):
        lines.append(f'  [sources: {len(r["sources"])} retrieved]')
    return '\n'.join(lines)
 
def main():
    ap = argparse.ArgumentParser(description='ANTIMONY Agent')
    ap.add_argument('--query', '-q', default=None, help='Single query mode')
    ap.add_argument('--json',  action='store_true', help='Output raw JSON')
    args = ap.parse_args()
 
    agent = Orchestrator()
 
    if args.query:
        result = agent.run(args.query)
        print(json.dumps(result, indent=2) if args.json else fmt_result(result))
        return
 
    print(BANNER)
    try:
        while True:
            query = input('Sb> ').strip()
            if not query: continue
            if query == '/help':
                print('  /help  /clear  /memory  /status  /exit')
                continue
            if query == '/clear':
                agent.memory.clear_session(); print('  Memory cleared.')
                continue
            if query == '/exit': break
            result = agent.run(query)
            print(fmt_result(result))
            print()
    except (KeyboardInterrupt, EOFError):
        print('\n  Session ended.')
 
if __name__ == '__main__':
    main()
