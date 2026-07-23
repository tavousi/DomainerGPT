import asyncio, aiodns

async def t():
    r = aiodns.DNSResolver(nameservers=["8.8.8.8","1.1.1.1"], timeout=6, tries=2)
    for d in ["google.com","cloudflare.com","openai.com"]:
        try:
            a = await r.query_dns(d, "A")
            print(d, "OK", len(a))
        except Exception as e:
            print(d, "ERR", repr(e))

asyncio.run(t())
