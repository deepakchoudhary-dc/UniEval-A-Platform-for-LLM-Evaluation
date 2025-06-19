# Security Policy

## 🔒 Protecting Your Privacy

This project is designed with privacy and security in mind. Please follow these guidelines:

### ⚠️ NEVER COMMIT SENSITIVE DATA

**DO NOT commit the following to GitHub:**
- ✅ Your actual `.env` file with real API keys
- ✅ Database files (*.db, *.sqlite)
- ✅ Log files containing user conversations
- ✅ Any files with personal information

### 🔑 API Key Security

1. **Use Environment Variables**: Store API keys in `.env` file (already gitignored)
2. **Use .env.example**: Template provided for setup instructions
3. **Rotate Keys**: Regularly rotate your API keys
4. **Limit Permissions**: Use API keys with minimal required permissions

### 🛡️ Deployment Security

When deploying to cloud platforms:
- Set environment variables in the platform's dashboard
- Enable HTTPS/SSL
- Use secrets management (AWS Secrets Manager, etc.)
- Monitor API usage and costs

### 🚨 If You Accidentally Commit Secrets

If you accidentally commit API keys or sensitive data:

1. **Immediately rotate/regenerate** the compromised keys
2. **Remove from git history** using:
   ```bash
   git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch .env' --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** to overwrite history:
   ```bash
   git push origin --force --all
   ```

### 📞 Reporting Security Issues

If you find security vulnerabilities in this project, please:
1. **DO NOT** open a public issue
2. Contact the maintainer privately
3. Provide detailed information about the vulnerability

### 🔍 Security Features

This chatbot includes:
- ✅ **Bias Detection**: Identifies potential bias in responses
- ✅ **Transparency**: Explainable AI with LIME/SHAP
- ✅ **Privacy-First**: No sensitive data in code
- ✅ **Secure Storage**: Environment variables for secrets
- ✅ **Audit Trail**: Comprehensive logging (without sensitive data)

### 🛠️ Security Best Practices

1. **Regular Updates**: Keep dependencies updated
2. **Input Validation**: Sanitize all user inputs
3. **Rate Limiting**: Implement API rate limiting
4. **Monitoring**: Monitor for unusual activity
5. **Backup**: Regular backups of non-sensitive data

Remember: **Security is everyone's responsibility!** 🔐
