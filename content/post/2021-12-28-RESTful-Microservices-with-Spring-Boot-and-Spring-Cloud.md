---
tags: [springboot]
title: RESTful Microservices with Spring Boot and Spring Cloud
created: '2022-04-18T00:38:46.309Z'
modified: '2022-04-26T22:27:30.593Z'
---

# RESTful Microservices with Spring Boot and Spring Cloud
- **RESTful web services with Spring Boot**
- **Microservices with Spring Cloud**

RESTful web services are the first step to developing microservices

1. Design RESTful web services with SpingBoot. Realize exception handling, validation, versioning, monitoring, visualization, documentation for services and resources. Connect services to database repo via JPA. HATEOAS (return uri for next possible actions, helps us get other info like operation on this resource and link of resource)

2. Establish communication between microservices, enable load balancing, scaling up and down of microservices. Centralize the configuration of microservices with Spring Cloud Config Server. Implement Eureka Naming Server and Feign to simplify connection and invocation between microservices

3. Bonus: Build containers for microservices with Docker and orchestrate microservices with Kubernetes

# 1.RESTful web services with SpringBoot
 ## Basic
* **Initialize with SPRING INITIALIZR**
* **Realize RESTful**
      @RestController on the top of Resource class
      @GetMapping(path = "") on the top of func
      @PathVariable in func args
      @Component on the head of DAOservice class(temp data and CRUD methods)
      @Autowired on the head of DAOservice object in Resource class
      @RequestBody in POST func args
* **Generic exception handling for all resources**
      @ExceptionHandler(Exception.class) to handle input request and exception and output error details(DIY timestamp message details) and HTTP_STATUS code
      In Resource class, only sth happens to fire exception throw
* **Validation for RESTful services**
      add new exception handler for validation
      add some limits like @Size @Past for the bean model
      add @Valid in RESTful func args

## Advanced
* **Generation of swagger documentation**
      swagger-ui.html is UI of OpenAPI in/v3/api-docs/ can present API and schema  
      new class SwaggerConfig
      @Configuration and @EnableSwagger2 on top of it
      define DEFAULT info
      @ApiModel on top of bean class, and @ApiModelProperty on top of each property
* **Monitor API with Actuator & Visualize API with HAL Explorer**
    add dependencies
* **Four ways to version API**
      @GetMapping("v1/person")
      @GetMapping(value = "/person/param", params ="version=1")
      @GetMapping(value = "/person/header", headers = "X-APIVERSION=1")
      @GetMapping(value = "/person/produces", produces = "application/vnd.company.app-v1+json")
## Connect to JPA
* **User Entity**
      reuse the User entity setup before
      introduce interface UserRepository extends JpaRepository
      @Repository on top of the interface
      replace DAOservice with UserRepository and @Autowired as well(JPA offer methods like find save so as to replace equilent methods in DAO)
      use sql script to add data into in-memory H2 database
* **Post Entity**
      @ManyToOne and  @JsonIgnore on top of user field
      introduce interface PostRepository extends JpaRepository
      @Repository on top of the interface
      add @OneToMany(mappedBy="user") to posts field in User entity
      add @Autowired PostRepository to UserResource

# 2.Microservices with SpringCloud
* **Get configuration from application property for service**
      ResourceController @Autowired Component who get props from application properties file
* **Set spring cloud config server to local git repo**
      configs for all microservice are centralized in a server
      @EnableConfigServer @SpringBootApplication on top of class SpringCloudConfigServerApplication
      set server port and git uri of it
      connect service to server by adding spring.cloud.config.uri=http://localhost:XXXX in its property file
* **Currency Exchange Microservice**
      Given two different currency, obtain the exchange rate between them
      In controller, find exchangeValue from JPA repo and set dynamic port for it
      In JPA repo, define findFromTo abstract func
      In entity, exchangeValue has from and to and exchageRate and port
* **Currency Conversion Microservice**
      Given two different currency and quantity, obtain the total amount of latter one
      In Bean, currencyConversion has from,to,exchangeRate,quantity,total,port fields
      In controller, use RestTemplate().getForEntity(uri of exchange service,class,vars) to statically invoke exchange service to get response
* **Use Feign to simplify invocation**
      In controller, @Autowired CurrencyExchangeServiceProxy proxy, and just simply proxy.retrieveExchangeValue(vars) to get response 
      In application, add @EnableFeignClients
      In CurrencyExchangeServiceProxy interface, add @FeignClient(name="currency-exchange-service") on the top and define the abstract func
* **Eureka Naming Server**
      new an application
      services register a name at name server, when invoked return address
      In application, add @EnableEurekaServer, add server port to property file
* **Connect conversion & exchange service to Eureka**
      add euraka dependency to pom.xml
      add euraka uri to property
      In application, add @EnableDiscoveryClient

