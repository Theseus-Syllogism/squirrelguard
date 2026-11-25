# Security

squirrel-guard handles video of people and, optionally, biometric templates. Please report security problems
privately rather than in a public issue.

**How to report:** use GitHub's private vulnerability reporting ("Security" tab -> "Report a vulnerability") on
this repository. Include what is affected, how to reproduce it, and what an attacker could do. Expect an answer
within 14 days. This is a volunteer project: there is no bug bounty.

**In scope:** the console (`./sg`, especially the local web console), the review interfaces, the evidence export
and verification code, the deployment files under `deploy/`, and anything that could expose video, faces, names,
credentials or evidence integrity.

**Known limitations** (documented, not vulnerabilities): MQTT without TLS on the local network; some internal hops
without certificate verification on loopback; the gaps listed in chapters 13 and 15 of the
[Feature and Capability Guide](FEATURE-GUIDE.pdf).

**Never include** real camera footage, faces, names, addresses or credentials in a report, an issue or a pull request.
