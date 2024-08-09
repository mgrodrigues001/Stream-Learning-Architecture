# Stream-Learning-Architecture
##Arquitetura de serviços orientados a Aprendizagem de Máquina Online.
###O MLOps tradicional muitas vezes não atende de forma eficaz à necessidade de atualizações contínuas de modelos de Machine Learning em fluxos de dados, especialmente em ambientes de produção. Este trabalho apresenta o desenvolvimento de uma arquitetura orientada a serviços, especificamente projetada para aprendizagem de máquina online. A arquitetura proposta utiliza uma combinação de ferramentas existentes e aplicações conteinerizadas, orquestradas por meio do Kubernetes, com o objetivo de otimizar a performance e a escalabilidade dos modelos de machine learning em ambientes de stream Learner. Nesta proposta, foram integradas ferramentas como o Mlflow para gerenciamento e rastreamento de modelos, Microk8s para orquestração dos contêiners, e API Flask para implementação das aplicações de inferência e atualização. A arquitetura foi avaliada de maneira extensiva em múltiplos aspectos, incluindo throughput do sistema, performance preditiva e impacto da frequência de atualização dos modelos. Os resultados destacam a importância de otimizar os parâmetros de atualização de modelos para equilibrar o throughput do sistema e a precisão preditiva, garantindo que os modelos permaneçam atualizados sem comprometer a capacidade de atender às solicitações de inferência de forma eficiente.
