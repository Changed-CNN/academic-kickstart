+++

title= "Build up personal website with Jekyll"
date=2019-08-24
tags=[ "markdown", "jekyll", "github-pages", "yaml", "vmware", "macos", "ruby", "domain"]

+++

## Status Quo

For most programmers today, the links that appear at the top of the search list when looking for professional-related questions in search engines are often those written by others on public blog like `CSDN`，`cnBlog`，`JianShu`，`BlogGarden`，but these are basically services provided to users by public blog sites, which are highly customized to some extent and do not allow users to customize. Although it has many advantages: **can find the content more accurately**, **can more easily to write blog**. It also **reduces the fun of the custom**, **makes the interface and plug-in  single**. At this time an exclusive personal website can meet the needs of a programmer who loves DIY, and an exclusive domain name will make it more cool. However, before there is a series of highly integrated website construction schemes, people need to master a series of development skills including front-end and back-end, which undoubtedly takes time and energy.
With the advent of `GithubPages+Jekyll`s solution on `Github`, it's going to be much easier, and now there are excellent static blogging frameworks for different platforms such as`Hexo`,here is an[ introduction to the differences between  Jekyll and Hexo](https://www.jianshu.com/p/ce1619874d34)，Since my personal website is based on`Jekyll`, so this blog only introduces `Jekyll` related content, in the future may launch other framework tutorial, please stay tuned.

## Operating Guidelines

- **Create a "Github Pages" project**
- **Configure The Environment Of Jekyll**
  - **VMware+macOS/Linux**
  - **Ruby**
  - **Mirroring（For Faster）**
  - **Jekyll**
  - **Bundle**
- **Theme**
- **Set Personal Domain**
  - **Create A CNAME**
  - **Purchase&Resolve Domain-Name**
  - **Flow Monitoring**
- **Detail Customization**
- **Write Blog**

## **Concrete Details**

### **Create a "Github Pages" project**

**What is "Github Pages"?**

> GitHub Pages is a static site hosting service designed to host your personal, organization, or project pages directly from a GitHub repository.
>

In plain English, it's a Github service that allows users to control their own static pages, either personal page or separate project page, by directly manipulating files in the repository.

- [Github Pages Official Introduction](https://help.github.com/en/articles/what-is-github-pages)

**How to create one?**

First of all, you need to have a Github account, if not, click [here](https://github.com/) and set up one for yourself

Then you need to create a `Repository`，the format of its name must be like`username.github.io`，and`username`is your account name

***Tip：The repository name must follow the exact format, otherwise it will not take effect***

If you are familiar with `git`, you can use `git` to manage your repository

If you are not，I think [Github Desktop](https://desktop.github.com/) may be a great choice，you can use a graphical interface to manage your projects more easily, and the history of changes to your projects can be easily reviewed

After the client logs in the account, you can find the created repository, clone the newly created repository to the local, and create a `index.html` file in the directory, roughly as follows:

```html
<!DOCTYPE html>
<html>
<body>
<h1>Hello World</h1>
<p>This is the first GitHub Pages</p>
</body>
</html>
```

Then click `commit to master`and `Fetch Origin` to synchronize it

Wait a while before you enter `username.github.io`in browser to access your own site

***Tip：Username here is the same as above, which is also your account name***

- [Github Pages Official Guidance](https://pages.github.com/)

### **Configure The Environment Of Jekyll**

**What is Jekyll?**

> `Jekyll` is a simple blog-like static site production machine.It has a template directory that contains documents in raw text format，converted to a full publishable static website with a converter（like [Markdown](http://daringfireball.net/projects/markdown/)）and [Liquid](https://github.com/Shopify/liquid/wiki) renderer that you can publish on any server you like。Jekyll also runs on `GitHub Pages`, which means that you can use`GitHub`to set up your project pages, blog or website

#### VMware+macOS/Linux

Since this step is not a key step, I will not elaborate on the specific operation of this part here

- If you want to install in the `macOS`enviroment, please refer to[VMware15+macOS10.13 installation guidance](https://blog.csdn.net/weixin_43299649/article/details/82881567)

- If you want to install in the `Linux`enviroment, please refer to[VMware12+CentOS installation guidance](https://blog.csdn.net/u010996565/article/details/79048104)

***Tip：It is recommended to install in Linux environment, although the installation effect is the same, but the apple virtual machine experience is very poor, no matter how much memory is allocated, no matter how many cores are allocated, there is still a clear sense of lag. If you have enough money, I suggest you to have a MacBook Pro and use the genuine macOS***

#### Ruby

`RVM`is a handy tool for managing and switching multiple versions of the`Ruby `environment. We need to enter the following instructions in the console

```
curl -sSL https://get.rvm.io | bash -s stable
```

After waiting a few minutes to install `RVM`, you can view the version number by following the following instruction

```
rvm -v
```

The version number appears to indicate successful installation, and it can be updated with the following instruction

```
rvm get stable
```

Next you can install and manage `Ruby`. The following instruction allows you to view versions of `Ruby` which can be installed currently

```
rvm list known
```

It is highly recommended that you install the latest version of `Ruby`, otherwise you will encounter a series of problems when you install the package later

```
rvm install 2.8.4
```

If you have many versions installed as I do, just set the default`Ruby` version with following instruction

```
rvm use 2.8.4 --default
```

The installation process is very long. After installation, you can use the following instruction to check the version. If the version number appears, the installation is successful

```
ruby -v
```

There are many similar installation guides on the web, and I refer to [here](https://www.jianshu.com/p/c073e6fc01f5)

#### Mirroring

`gem` is a package manager for `ruby`, just like`pip` of `python`

So `gem` also has the corresponding mirror source, if you feel that using the default mirror is too slow, you can use the following instruction to add a Chinese domestic mirror

```
gem source -a http://gems.ruby-china.com/ 
```

The following instruction allows you to view the mirror currently in use

```
gem source
```

#### Jekyll

The best way to install `Jekyll` is to use `RubyGems`with the following instruction

```
gem install jekyll
```

All `Jekyll` dependency packages are automatically installed, no worries, and can be updated with the following instruction

```
gem update jekyll
```

Please refer to [the official Jekyll installation documentation](http://jekyllcn.com/docs/installation/) for more details

#### Bundle

`Bundle` is a powerful tool for configuring the environment. It only needs a `Gemfile`configuration file to configure the corresponding environment. It can be installed through the following instruction

```
gem install bundle
```

Go to the local repository directory`/Users/XXX/Documents/git/XXX.github.io`, and execute the following instructions to activate the environment

```
bundle install
bundle exec jekyll serve
```

The following information indicates success

```
Server address: http://127.0.0.1:4000/
Server running... press ctrl-c to stop.
```

- Local validation：Use a browser to visit`http://127.0.0.1:4000/`to view the locally generated website

- Online validation：Click `commit to master` on `Github` client, then click `Fetch Origin` to synchronize, wait a while and enter `username.github.io` in the browser to access the website

The distinction of `Ruby`'s related concepts can refer to  [bundler vs RVM vs gems vs RubyGems vs gemsets vs system ruby ](https://stackoverflow.com/questions/15586216/bundler-vs-rvm-vs-gems-vs-rubygems-vs-gemsets-vs-system-ruby)

The concept of `Jekyll` and the installation process can refer to [Jekyll中文网站](http://jekyllcn.com/)and[Jekyll English website](https://jekyllrb.com/)

### Theme 

There are many well-designed themes to choose from [Official theme website](http://jekyllthemes.org/)

Select one and download it locally and put it directly into your `username.github.io` repository，then it can be viewed after synchronization

The `readme.md` file for each theme details its specific use

My website is using a theme called [HardCandy](http://jekyllthemes.org/themes/HardCandy-Jekyll/)，thanks to the author who produced the theme

### Set Personal Domain

#### Create A CNAME

There are two ways to create：

- Open the repository page in the browser, click `setting` at the upper right corner, slide down to `Github Pages` section, fill in your personal domain name in`Custom domain`, and save it to generate automatically

- Click `Create new file`in the repository，then write the name as `CNAME` without the suffix, and add up a naked domain name without the protocol name, instead of beginning with `http://`, like `fengzhikang.xyz`

Refer to[ the official custom domain name configuration](https://help.github.com/en/articles/adding-or-removing-a-custom-domain-for-your-github-pages-site)

#### Purchase&Resolve Domain-Name

You can purchase one in[wanwang](https://wanwang.aliyun.com/)，The price at first year is usually very low, the best strategy is to change the domain name every year.

Enter `domain name console` after the purchase is completed, then click `parse` and `add record`, waiting for the effect, you can check the status through the following instruction

```
dig joe-liu.com +nostats +nocomments +nocmd
```

#### Flow Monitoring

Enter[Baidu Web Statistics](https://tongji.baidu.com/web/10000047685/welcome/login), register an account, `add a new website` in `management`, fill in `domain name` and `homepage` and a piece of code will appear later. Paste this code into `head.html` in `_include` folder in the repository, wait for 20 minutes before it takes effect, and then you can view the traffic report in the homepage.

### Detail Customization

First we need to understand the concept interpretation and directory structure of Jekyll

>At the heart of Jekyll is a text conversion engine.The idea is that you can write in your favorite markup language, whether it's Markdown, Textile, or simple HTML, and Jekyll will take you through a layout or series of layouts.Throughout the process you can set the URL path, how your text will appear in the layout, and so on.This can be done through plain text editing, and the resulting static page is your result.

A basic Jekyll site usually looks something like this:

```
.
├── _config.yml
├── _drafts
|   ├── begin-with-the-crazy-ideas.textile
|   └── on-simplicity-in-technology.markdown
├── _includes
|   ├── footer.html
|   └── header.html
├── _layouts
|   ├── default.html
|   └── post.html
├── _posts
|   ├── 2007-10-29-why-every-programmer-should-play-nethack.textile
|   └── 2009-04-26-barcamp-boston-4-roundup.textile
├── _site
├── .jekyll-metadata
└── index.html
```

An overview of what each of these does:

| FILE / DIRECTORY                                          | DESCRIPTION                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| `_config.yml`                                             | Stores [configuration](https://jekyllrb.com/docs/configuration/) data. Many of these options can be specified from the command line executable but it’s easier to specify them here so you don’t have to remember them. |
| `_drafts`                                                 | Drafts are unpublished posts. The format of these files is without a date: `title.MARKUP`. Learn how to [work with drafts](https://jekyllrb.com/docs/posts/#drafts). |
| `_includes`                                               | These are the partials that can be mixed and matched by your layouts and posts to facilitate reuse. The liquid tag can be used to include the partial in`_includes/file.ext`. |
| `_layouts`                                                | These are the templates that wrap posts. Layouts are chosen on a post-by-post basis in the [front matter](https://jekyllrb.com/docs/front-matter/), which is described in the next section. The liquid tag is used to inject content into the web page. |
| `_posts`                                                  | Your dynamic content, so to speak. The naming convention of these files is important, and must follow the format:`YEAR-MONTH-DAY-title.MARKUP`. The [permalinks](https://jekyllrb.com/docs/permalinks/) can be customized for each post, but the date and markup language are determined solely by the file name. |
| `_data`                                                   | Well-formatted site data should be placed here. The Jekyll engine will autoload all data files (using either the `.yml`, `.yaml`, `.json`, `.csv` or `.tsv` formats and extensions) in this directory, and they will be accessible via `site.data`. If there's a file `members.yml` under the directory, then you can access contents of the file through `site.data.members`. |
| `_sass`                                                   | These are sass partials that can be imported into your `main.scss`which will then be processed into a single stylesheet `main.css`that defines the styles to be used by your site. |
| `_site`                                                   | This is where the generated site will be placed (by default) once Jekyll is done transforming it. It’s probably a good idea to add this to your `.gitignore` file. |
| `.jekyll-metadata`                                        | This helps Jekyll keep track of which files have not been modified since the site was last built, and which files will need to be regenerated on the next build. This file will not be included in the generated site. It’s probably a good idea to add this to your`.gitignore` file. |
| `index.html` or `index.md` and other HTML, Markdown files | Provided that the file has a [front matter](https://jekyllrb.com/docs/front-matter/) section, it will be transformed by Jekyll. The same will happen for any `.html`, `.markdown`, `.md`, or `.textile` file in your site’s root directory or directories not listed above. |
| Other Files/Folders                                       | Every other directory and file except for those listed above—such as`css` and `images` folders, `favicon.ico` files, and so forth—will be copied verbatim to the generated site. There are plenty of [sites already using Jekyll](https://jekyllrb.com/showcase/) if you’re curious to see how they’re laid out. |

Through some experimentation, I've found that there aren't many files that we can work with directly as a user

- The folder storing `source` file，including pictures, icons, etc.
- Each`HTML` file corresponds to the`page`，and `index.html` is the home page
- The`_layouts`folder  controls the layout，including formats like`post`and`default`
- The`_posts`folder stores `blog`，including `markdown` file
- The most important is `_config.yml`which configures your site most intuitively

### Write Blog

`Blog`can be written in text markup languages like`markdown`，`textile`，`html`

If you also like using `markdown`，I highly recommend a tool named [Typora](https://www.typora.io/)，because it can really preview in real time

As for the specific grammar of `markdown`, I will write a separate article to summarize , please stay tuned

At the head of every `blog` we write we need to add a piece of`yaml` code(configuration item content is customized)

```yaml
layout: post
title:  "Build up personal website with Jekyll"
subtitle: 'Posted by Zhikang Feng'
date:   2019-08-24
tags: markdown jekyll github-pages yaml vmware macos ruby domain
description: 'Posted by Zhikang Feng'
color: 'rgb(154,253,55)'
cover: '/assets/profile.jpeg'
```

`YAML`'s' header information can be configured refer to [here](http://jekyllcn.com/docs/frontmatter/)

After saving, a file named`YEAR-MONTH-DAY-TITLE.md`will be automatically generated，and put it into the directory`_posts`  to be synchronized to take effect

## Summary

> This blog is my first technical blog. There may be some minor details that are not in place in my initial creation. I hope you can point them out in the comments. I also hope this article can help you in the process of building your personal website. In the future, I will gradually improve the previous articles to make them more understandable and targeted. Thank you!