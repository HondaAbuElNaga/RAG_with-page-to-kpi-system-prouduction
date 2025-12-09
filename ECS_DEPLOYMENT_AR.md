# دليل رفع التطبيق على AWS ECS مع EFS

## المتطلبات الأساسية

1. AWS CLI مثبت ومضبوط
2. Docker image مرفوع على Docker Hub أو ECR
3. EFS file system موجود في AWS
4. ECS cluster موجود

## الخطوات

### 1. التأكد من إعدادات EFS

تأكد من:
- EFS موجود في نفس VPC الخاص بـ ECS cluster
- Security Group الخاص بـ EFS يسمح بـ NFS traffic (port 2049) من ECS security group
- File System ID: `fs-082c090d69c139f7b` (موجود في task definition)

### 2. تسجيل Task Definition

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

### 3. إنشاء أو تحديث ECS Service

```bash
aws ecs create-service \
  --cluster your-cluster-name \
  --service-name sstli-chatbot-service \
  --task-definition sstli-chatbot-v3 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

أو لتحديث service موجود:

```bash
aws ecs update-service \
  --cluster your-cluster-name \
  --service sstli-chatbot-service \
  --task-definition sstli-chatbot-v3 \
  --force-new-deployment
```

### 4. التحقق من الصلاحيات

تأكد من أن `ecsTaskExecutionRole` لديه الصلاحيات التالية:
- `elasticfilesystem:ClientMount`
- `elasticfilesystem:ClientWrite`
- `elasticfilesystem:ClientRootAccess`

## ملاحظات مهمة

### المسارات في Container:
- **EFS Mount Point**: `/data` (يتم mount من EFS)
- **ChromaDB Path**: `/data/chroma_db` (يتم حفظه في EFS)
- **Database Path**: `/data/kpi_data.db` (يتم حفظه في EFS)

### Environment Variables:
- `CHROMA_PATH=/data/chroma_db` - مسار ChromaDB في EFS
- `DB_PATH=/data/kpi_data.db` - مسار قاعدة البيانات في EFS
- `OPENAI_API_KEY` - مفتاح OpenAI API
- `ADMIN_USER` و `ADMIN_PASS` - بيانات الدخول للإدارة

### الأمان (مهم جداً):

**⚠️ تحذير**: الـ API keys موجودة مباشرة في task definition. يُفضل استخدام AWS Secrets Manager:

1. إنشاء secrets في AWS Secrets Manager:
```bash
aws secretsmanager create-secret \
  --name rag-app/secrets \
  --secret-string '{"OPENAI_API_KEY":"your-key","ADMIN_USER":"admin","ADMIN_PASS":"password"}'
```

2. تحديث task definition لاستخدام secrets بدلاً من environment variables.

## استكشاف الأخطاء

### المشكلة: Container لا يستطيع الكتابة على EFS
**الحل**: 
- تحقق من Security Groups (يجب السماح بـ port 2049)
- تحقق من صلاحيات IAM role
- تحقق من أن EFS في نفس VPC

### المشكلة: البيانات لا تظهر
**الحل**:
- تأكد من أن المسارات صحيحة (`/data/chroma_db` و `/data/kpi_data.db`)
- تحقق من logs: `aws logs tail /ecs/sstli-chatbot --follow`

### المشكلة: Container لا يبدأ
**الحل**:
- تحقق من CloudWatch Logs
- تأكد من أن Docker image موجود وصحيح
- تحقق من أن جميع environment variables موجودة

## تحديث الكود

بعد أي تغييرات في الكود:

1. بناء Docker image جديد:
```bash
cd image
docker build -t ebrahemhesham/rag-app:v1 .
docker push ebrahemhesham/rag-app:v1
```

2. تحديث service:
```bash
aws ecs update-service \
  --cluster your-cluster-name \
  --service sstli-chatbot-service \
  --force-new-deployment
```

## البيانات المستمرة

- جميع البيانات (ChromaDB + SQLite) تُحفظ في EFS
- البيانات تبقى حتى بعد إعادة تشغيل أو حذف containers
- البيانات مشتركة بين جميع instances من نفس service

